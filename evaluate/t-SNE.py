import matplotlib.pyplot as plt
import sklearn.manifold as manifold
import Celeb.data_loader
from models.ECNet import EmbeddingCapsNet
import argparse
import torch
import os
import ProcessingTools as pt
import json
import torchvision
import models.ECNet
import random
import numpy as np


def extract_feature(model, query_loader, gallery_loader, selected_pids):
    features = list()
    pids = list()

    # print('\nextracting query features')
    # for i, inputs in pt.ProgressBar(enumerate(query_loader), finish_mark=None, max=len(query_loader)):
    #     model.eval()
    #     with torch.no_grad():
    #         imgs, pid, _ = inputs
    #         data = skip_pid(imgs, pid, selected_pids)
    #
    #         if data:
    #             imgs, pid = data
    #             if torch.cuda.is_available():
    #                 imgs = imgs.cuda()
    #
    #             _, _, (outputs, feature) = model(imgs)
    #             mean_pool = torch.reshape(torch.nn.AvgPool2d(feature.shape[2:4])(feature), feature.shape[0:2])
    #             outputs = torch.cat((models.ECNet.l2(outputs), mean_pool * 0.55), dim=1)
    #
    #             imgs_f = torchvision.transforms.RandomHorizontalFlip(p=1)(imgs)
    #             _, _, (outputs_f, feature_f) = model(imgs_f)
    #             mean_pool_f = torch.reshape(torch.nn.AvgPool2d(feature_f.shape[2:4])(feature_f), feature_f.shape[0:2])
    #             outputs = (torch.cat((models.ECNet.l2(outputs_f), mean_pool_f * 0.55), dim=1) + outputs) / 2
    #
    #             outputs = outputs.data.cpu()
    #             features.append(outputs)
    #             pids.append(pid)

    print('\nextracting gallery features')
    for i, inputs in pt.ProgressBar(enumerate(gallery_loader), finish_mark=None, max=len(gallery_loader)):
    # for i, inputs in pt.ProgressBar(enumerate(gallery_loader), finish_mark=None, max=4):
        model.eval()
        with torch.no_grad():
            imgs, pid, _ = inputs
            data = skip_pid(imgs, pid, selected_pids)

            if data:
                imgs, pid = data
                if torch.cuda.is_available():
                    imgs = imgs.cuda()

                _, _, (outputs, feature) = model(imgs)
                mean_pool = torch.reshape(torch.nn.AvgPool2d(feature.shape[2:4])(feature), feature.shape[0:2])
                outputs = torch.cat((models.ECNet.l2(outputs), mean_pool * 0.55), dim=1)

                imgs_f = torchvision.transforms.RandomHorizontalFlip(p=1)(imgs)
                _, _, (outputs_f, feature_f) = model(imgs_f)
                mean_pool_f = torch.reshape(torch.nn.AvgPool2d(feature_f.shape[2:4])(feature_f), feature_f.shape[0:2])
                outputs = (torch.cat((models.ECNet.l2(outputs_f), mean_pool_f * 0.55), dim=1) + outputs) / 2

                outputs = outputs.data.cpu()
                features.append(outputs)
                pids.append(pid)

    return torch.cat(features, dim=0), torch.cat(pids, dim=0)


def skip_pid(imgs, pids, selected_pids):
    img_list = list()
    pid_list = list()

    for img, pid in zip(imgs, pids):
        if pid in selected_pids:
            img_list.append(img)
            pid_list.append(pid)

    if len(img_list) == 0: return False
    else: return torch.stack(img_list, dim=0), torch.stack(pid_list, dim=0)


def main(args):
    with open(args.configs, 'r') as f:
        cf = json.load(f)['configs']

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

    gallery_loader = Celeb.data_loader.get_loader(f'{args.data_root}/gallery', height, width, relabel=False,
                                                  batch_size=args.batch_size, mode='test', num_workers=args.workers,
                                                  name_pattern='celeb')
    query_loader = Celeb.data_loader.get_loader(f'{args.data_root}/query', height, width, relabel=False,
                                                batch_size=args.batch_size, mode='test', num_workers=args.workers,
                                                name_pattern='celeb')

    # all_pids = list()
    # for _, pid, _ in pt.ProgressBar(gallery_loader.dataset, max=len(gallery_loader.dataset)):
    #     if pid not in all_pids:
    #         all_pids.append(pid)
    # selected_ids = random.choices(all_pids, k=args.id_num) if args.id_num >= 0 else all_pids
    selected_ids = [3, 18, 20, 2, 59, 39, 13, 17, 20, 69, 72]  # LTCC
    # selected_ids = [263, 237, 166, 311, 155, 234, 193, 59, 148, 9, 267]  # celeb
    # selected_ids = [44, 56, 74, 47, 53, 3, 56, 63, 72, 21, 42]

    print(f'PID: {selected_ids}')

    features, pids = extract_feature(model, query_loader, gallery_loader, selected_ids)

    tsneNDArray = manifold.TSNE(n_components=2, init="pca", random_state=0, n_iter=10000).fit_transform(features)

    figure, axesSubplot = plt.subplots()

    pids_color = np.array(pids)
    pids = np.array(pids)
    for i in range(len(selected_ids)):
        pids_color[pids == selected_ids[i]] = i

    # temp = axesSubplot.scatter(tsneNDArray[:, 0], tsneNDArray[:, 1], c=pids_color, cmap='nipy_spectral', label=pids_color, s=40)
    temp = axesSubplot.scatter(tsneNDArray[:, 0], tsneNDArray[:, 1], c=pids_color, cmap='nipy_spectral', label=pids_color, s=15)

    axesSubplot.set_xticks(())
    axesSubplot.set_yticks(())

    for axis in ['top', 'bottom', 'left', 'right']:
        axesSubplot.spines[axis].set_linewidth(6)
        axesSubplot.spines[axis].set_zorder(0)

    # legend1 = axesSubplot.legend(*temp.legend_elements(), loc="lower left", title="Person", ncol=11)
    # axesSubplot.add_artist(legend1)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./Celeb-reID-light')
    parser.add_argument('--configs', type=str, default='./configs/Celeb-reID-light.json')
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default=None, help='If let None, GPU set automatically.')
    parser.add_argument('--id_num', type=int, default=11, help='The number of ids.')
    args = parser.parse_args()

    # set gpu
    if args.gpu_id: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # modify relative path to absolute path
    args.data_root = os.path.abspath(args.data_root)

    main(args)
