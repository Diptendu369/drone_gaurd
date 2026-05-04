import pprint
import argparse
import tqdm
import os
import csv
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
import Datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models.model import DroneGuard as get_model
import time

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# --cfg experiments/sha/sha_wresnet.yaml --model-file output/shanghai/sha_wresnet/shanghai.pth GPUS [3]
# --cfg experiments/ped2/ped2_wresnet.yaml --model-file output/ped2/ped2_wresnet/ped2.pth GPUS [3]
def parse_args():
    parser = argparse.ArgumentParser(description='Test Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='pretrained/shanghaitech.pth', type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False       # TODO ? False
    config.freeze()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    gpus = [(config.GPUS[0])] if use_cuda else []
    model = get_model(config)
    logger.info('Model: {}'.format(model.get_name()))
    if use_cuda:
        model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])
    logger.info('Epoch: '.format(args.model_file))

    
    # load model
    state_dict = torch.load(args.model_file, map_location=device)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        if isinstance(model, nn.DataParallel):
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    test_dataset = eval('Datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * (len(gpus) if use_cuda else 1),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    
    if config.DATASET.TYPE == "ground":
        mat=anomaly_util.get_labels(config.DATASET.DATASET)
    else :
        mat=anomaly_util.get_drone_labels(config.DATASET.DATASET)
    psnr_list = inference(config, test_loader, model,args)
    assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

    auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)

    # Visualization and exports
    try:
        results_dir = os.path.join('results', config.DATASET.DATASET)
        os.makedirs(results_dir, exist_ok=True)

        # Save ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={auc*100:.1f}%')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {config.DATASET.DATASET}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
        plt.close()

        # Save PSNR as numpy and CSV, and plot per-video curves
        import numpy as _np
        _np.save(os.path.join(results_dir, 'psnr_list.npy'),
                 _np.array([_np.array(v, dtype=_np.float32) for v in psnr_list], dtype=object),
                 allow_pickle=True)

        # CSV (ragged rows padded with blanks)
        csv_path = os.path.join(results_dir, 'psnr_list.csv')
        max_len = max(len(v) for v in psnr_list) if len(psnr_list) > 0 else 0
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['video_index'] + [f'frame_{i}' for i in range(max_len)]
            writer.writerow(header)
            for i, v in enumerate(psnr_list):
                row = [i] + v + ["" for _ in range(max_len - len(v))]
                writer.writerow(row)

        for i, v in enumerate(psnr_list):
            plt.figure()
            plt.plot(v)
            plt.xlabel('Frame index')
            plt.ylabel('PSNR')
            plt.title(f'PSNR - video {i}')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'psnr_video_{i:03d}.png'))
            plt.close()
    except Exception as e:
        logger.info(f"Visualization export skipped due to error: {e}")

    logger.info(f'AUC: {auc * 100:.1f}% ')


def inference(config, data_loader, model,args,quiet=False):
    loss_func_mse = nn.MSELoss(reduction='none')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.eval()
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if(not(quiet)):
              print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []
            # compute the output
            video, video_name = train_util.decode_input(input=data, train=False)
            video = [frame.to(device=device) for frame in video]
            for f in tqdm.tqdm(range(len(video) - fp),disable=quiet):
                inputs = video[f:f + fp]
                model_run = model.module if isinstance(model, nn.DataParallel) else model
                output = model_run(inputs)
                target = video[f + fp:f + fp + 1][0]

                # compute PSNR for each frame
                # https://github.com/cvlab-yonsei/MNAD/blob/d6d1e446e0ed80765b100d92e24f5ab472d27cc3/utils.py#L20
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)
           
    return psnr_list 


if __name__ == '__main__':
    main()

