import Celeb.evaluate
import Celeb.data_loader
from models.ECNet import EmbeddingCapsNet
import argparse
import torch
import os
import ProcessingTools as pt
import json


def reIDeval(args):
    with open(args.configs, 'r') as f:
        cf = json.load(f)['configs']
    pt.create_folder(args.save_path)
    logs = open(f'{args.save_path}/eval.txt', 'a')

    height = cf['height']
    width = cf['width']
    classes_num = cf['n_class']

    # load model
    model = EmbeddingCapsNet(num_classes=classes_num, ratio=cf['ratio'], cuda=torch.cuda.is_available())
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint)

    if torch.cuda.device_count() > 1:
        print(f'{torch.cuda.device_count()} GPUs are detected.')
        model = torch.nn.DataParallel(model.cuda())
    elif torch.cuda.is_available():
        print('1 GPU is detected.')
        model = model.cuda()
    else: print('run on cpu.')

    evaluator = Celeb.evaluate.Evaluator(model)
    gallery_loader = Celeb.data_loader.get_loader(f'{args.data_root}/gallery', height, width, relabel=False,
                                                  batch_size=args.batch_size, mode='test', num_workers=args.workers,
                                                  name_pattern='celeb')
    query_loader = Celeb.data_loader.get_loader(f'{args.data_root}/query', height, width, relabel=False,
                                                batch_size=args.batch_size, mode='test', num_workers=args.workers,
                                                name_pattern='celeb')

    evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.ret, gallery_loader.dataset.ret, logs, args.n_rerank, cf['re_rank']['k1'], cf['re_rank']['k2'], cf['re_rank']['lambda'])
    logs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./Celeb-reID-light')
    parser.add_argument('--configs', type=str, default='./configs/Celeb-reID-light.json')
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default=None, help='If let None, GPU set automatically.')
    parser.add_argument('--save_path', type=str, default='./evaluate/outputs')
    parser.add_argument('--n_rerank', action='store_false', default=True)
    args = parser.parse_args()

    # set gpu
    if args.gpu_id: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # modify relative path to absolute path
    args.data_root = os.path.abspath(args.data_root)

    reIDeval(args)
