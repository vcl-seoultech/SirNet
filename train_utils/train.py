import torch
import Celeb.data_loader
from models.ECNet import EmbeddingCapsNet
import train_utils.trainer
import Celeb.evaluate
import argparse
import ProcessingTools as pt
import os
import shutil
import json
import random
import numpy as np


def train(args):
    with open(args.configs, 'r') as f:
        cf = json.load(f)['configs']

    # record the environments
    pt.create_folder(args.save_path)
    pt.create_folder(f'{args.save_path}/recon')
    config_name = args.configs.split('\\')[-1] if '\\' in args.configs else args.configs.split('/')[-1]
    shutil.copy(args.configs, f'{args.save_path}/{config_name}')
    shutil.copytree('./', f'{args.save_path}/snapshot/') if args.snapshot else None
    logs = open(f'{args.save_path}/logs.txt', 'a')
    comments = input('Comments for the logs file: ') if args.auto_comment is None else args.auto_comment
    print(f'Comments: {comments}\n', file=logs)

    with open(f'{args.save_path}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    height = cf['height']
    width = cf['width']
    classes_num = cf['n_class']

    # load data loader
    train_loader = Celeb.data_loader.get_loader(f'{args.data_root}/train', height, width, relabel=True, batch_size=args.batch_size, mode='train', num_workers=args.workers, name_pattern='celeb')
    all_loader = Celeb.data_loader.get_loader(f'{args.data_root}/train', height, width, relabel=True, batch_size=args.batch_size * 5, mode='test', num_workers=args.workers, name_pattern='celeb')
    gallery_loader = Celeb.data_loader.get_loader(f'{args.data_root}/gallery', height, width, relabel=False, batch_size=args.batch_size, mode='test', num_workers=args.workers, name_pattern='celeb')
    query_loader = Celeb.data_loader.get_loader(f'{args.data_root}/query', height, width, relabel=False, batch_size=args.batch_size, mode='test', num_workers=args.workers, name_pattern='celeb')

    model = EmbeddingCapsNet(num_classes=classes_num, ratio=cf['ratio'], cuda=torch.cuda.is_available())

    # load model if it is needed
    if args.start_epoch > 0 and args.model:
        print('train resume')
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint)
        del checkpoint

    # check GPU env
    if torch.cuda.device_count() > 1:
        print(f'{torch.cuda.device_count()} GPUs are detected.')
        model = torch.nn.DataParallel(model.cuda())
    elif torch.cuda.is_available():
        print('1 GPU is detected.')
        model = model.cuda()
    else: print('run on cpu.')

    model = train_utils.trainer.train(model, train_loader, all_loader, args.start_epoch, args.epoch, args.save_path, cf, logs, torch.cuda.is_available())

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), f'{args.save_path}/model_{args.epoch:04d}.pth.tar')
    else:
        torch.save(model.state_dict(), f'{args.save_path}/model_{args.epoch:04d}.pth.tar')

    # evaluate model after training
    logs = open(logs.name, 'a')
    evaluator = Celeb.evaluate.Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.ret, gallery_loader.dataset.ret, logs, args.n_rerank, cf['re_rank']['k1'], cf['re_rank']['k2'], cf['re_rank']['lambda'])
    logs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./Celeb-reID-light')
    parser.add_argument('--configs', type=str, default='./configs/Celeb-reID-light.json')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='./outputs')
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--gpu_id', type=str, default=None)
    parser.add_argument('--auto_comment', type=str, default=None)
    parser.add_argument('--snapshot', action='store_true', default=False)
    parser.add_argument('--n_rerank', action='store_false', default=True)

    # parameter for resume
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()

    # set gpu
    if args.gpu_id: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # modify relative path to absolute path
    args.save_path = os.path.abspath(args.save_path)
    args.data_root = os.path.abspath(args.data_root)
    args.configs = os.path.abspath(args.configs)

    # set random seed
    torch.manual_seed(486)
    random.seed(486)
    np.random.seed(486)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.cuda.manual_seed(486)
    # torch.cuda.manual_seed_all(486)  # if use multi-GPU

    # train_utils start
    train(args)
