import Celeb.data_loader
from models.ECNet import EmbeddingCapsNet
import argparse
import torch
import os
import ProcessingTools as pt
import json
import Celeb.evaluate
import random
import ProcessingTools.torch_tools as tt
import numpy as np
import cv2


def visualize_mAP(args):
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
    else:
        print('run on cpu.')

    evaluator = Celeb.evaluate.Evaluator(model)
    gallery_loader = Celeb.data_loader.get_loader(f'{args.data_root}/gallery', height, width, relabel=False,
                                                  batch_size=args.batch_size, mode='test', num_workers=args.workers,
                                                  name_pattern='celeb')
    query_loader = Celeb.data_loader.get_loader(f'{args.data_root}/query', height, width, relabel=False,
                                                batch_size=args.batch_size, mode='test', num_workers=args.workers,
                                                name_pattern='celeb')

    _, dismat = evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.ret, gallery_loader.dataset.ret, logs, args.n_rerank, cf['re_rank']['k1'], cf['re_rank']['k2'], cf['re_rank']['lambda'])
    logs.close()

    # all_query = [x for x in range(len(query_loader.dataset))]
    # selected_query = random.choices(all_query, k=args.candidate)
    # print(selected_query)
    # selected_query = [36, 1567, 390, 1237]     # Celeb
    selected_query = [187, 265, 332, 491]  # LTCC_CC
    # selected_query = [38, 173, 305, 125, 487, 149, 318, 67, 335, 112, 162, 177, 226, 88, 17, 491, 234, 16, 258, 74]

    index = list()
    for query in selected_query:
        dis = dismat[query]
        sort_dis = sorted(dis)
        index.append([dis.tolist().index(sort_dis[x]) for x in range(args.rank)])

    mAP_imgs = list()
    query_imgs = list()
    for n, query in enumerate(selected_query):
        q_img, q_pid, _ = query_loader.dataset[query]
        query_imgs.append(np.array(tt.torch_img_denormalize(q_img).permute(1, 2, 0)).astype('uint8')[:, :, ::-1])

        gallery_imgs = list()
        for gallery in index[n]:
            g_img, g_pid, _ = gallery_loader.dataset[gallery]
            g_img = np.array(tt.torch_img_denormalize(g_img).permute(1, 2, 0)).astype('uint8')[:, :, ::-1]

            g_img = cv2.rectangle(g_img.copy(), (0, 0), (128, 256), (0, 255, 0), 9) if g_pid == q_pid else cv2.rectangle(g_img.copy(), (0, 0), (128, 256), (0, 0, 255), 9)
            gallery_imgs.append(g_img)

        mAP_imgs.append(np.concatenate(gallery_imgs, axis=1))

    cv2.imwrite(f'{args.save_path}/query.png', np.concatenate(query_imgs, axis=0))
    cv2.imwrite(f'{args.save_path}/mAP.png', np.concatenate(mAP_imgs, axis=0))


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
    parser.add_argument('--candidate', type=int, default=20)
    parser.add_argument('--rank', type=int, default=4)
    args = parser.parse_args()

    # set gpu
    if args.gpu_id: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # modify relative path to absolute path
    args.data_root = os.path.abspath(args.data_root)

    visualize_mAP(args)