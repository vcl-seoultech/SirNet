import Celeb.evaluate
import argparse
import os
from models.ECNet import EmbeddingCapsNet
import torch
import json
import ProcessingTools as pt
import Celeb.data_loader
import numpy as np
import ProcessingTools.torch_tools as tt
import cv2


def dis(model, query_loader, gallery_loader):
    query_features, _ = Celeb.evaluate.extract_features(model, query_loader, 1)
    gallery_features, _ = Celeb.evaluate.extract_features(model, gallery_loader, 1)
    return Celeb.evaluate.pairwise_distance(query_features, gallery_features, query_loader.dataset.ret, gallery_loader.dataset.ret)


def cal_mAP(model, query_loader, gallery_loader):
    dist_mat = dis(model, query_loader, gallery_loader).cpu().numpy()

    query_ids = np.array([pid for _, pid, _ in query_loader.dataset.ret])
    gallery_ids = np.array([pid for _, pid, _ in gallery_loader.dataset.ret])

    indices = np.argsort(dist_mat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    AP = (np.cumsum(matches, axis=1) * matches) / np.linspace(1, dist_mat.shape[1], dist_mat.shape[1]) / \
         np.sum(matches, axis=1)[..., None]
    return np.sum(AP, axis=1) * 100, query_ids, gallery_ids


def save_img(model, q_max_img, q_min_img, g_max_img, g_min_img, save_path):
    q_max_img = torch.cat((torch.unsqueeze(q_max_img, dim=0), g_max_img))
    q_min_img = torch.cat((torch.unsqueeze(q_min_img, dim=0), g_min_img))

    class_weight = model.classifier_activation if torch.cuda.device_count() <= 1 else model.module.classifier_activation

    model.eval()
    with torch.no_grad():
        (_, _, pid_max), _, max_att = model(q_max_img.cuda())
        (_, _, pid_min), _, min_att = model(q_min_img.cuda())

    pid_max = torch.argmax(pid_max, dim=1)
    pid_min = torch.argmax(pid_min, dim=1)

    max_att = class_weight.weight[pid_max][..., None, None] * max_att[1]
    min_att = class_weight.weight[pid_min][..., None, None] * min_att[1]

    max_att = torch.nn.functional.interpolate(torch.sum(max_att.cpu(), dim=1, keepdim=True), q_max_img.shape[2:], mode='bilinear', align_corners=False)
    min_att = torch.nn.functional.interpolate(torch.sum(min_att.cpu(), dim=1, keepdim=True), q_max_img.shape[2:], mode='bilinear', align_corners=False)

    q_max_img = torch.unsqueeze(torch.cat([_ for _ in q_max_img], dim=2), dim=0)
    q_min_img = torch.unsqueeze(torch.cat([_ for _ in q_min_img], dim=2), dim=0)
    max_att = torch.unsqueeze(torch.cat([_ for _ in max_att], dim=2), dim=0)
    min_att = torch.unsqueeze(torch.cat([_ for _ in min_att], dim=2), dim=0)

    tt.torch_imgs_save(q_max_img, f'{save_path}/max')
    tt.torch_imgs_save(q_min_img, f'{save_path}/min')

    q_max_img = np.array(tt.torch_img_denormalize(q_max_img)[0].permute(1, 2, 0)).astype('uint8')
    q_min_img = np.array(tt.torch_img_denormalize(q_min_img)[0].permute(1, 2, 0)).astype('uint8')
    max_att = cv2.applyColorMap(np.array(tt.torch_img_denormalize(max_att)[0].permute(1, 2, 0).detach()).astype('uint8'), cv2.COLORMAP_JET)
    min_att = cv2.applyColorMap(np.array(tt.torch_img_denormalize(min_att)[0].permute(1, 2, 0).detach()).astype('uint8'), cv2.COLORMAP_JET)

    q_max_img = cv2.cvtColor(q_max_img[:, :, ::-1], cv2.COLOR_BGR2GRAY)[..., None]
    q_min_img = cv2.cvtColor(q_min_img[:, :, ::-1], cv2.COLOR_BGR2GRAY)[..., None]

    cv2.imwrite(f'{save_path}/max_att.png', (q_max_img[:, :, ::-1] / 2 + max_att / 2))
    cv2.imwrite(f'{save_path}/min_att.png', (q_min_img[:, :, ::-1] / 2 + min_att / 2))


def visualize_feature(args):
    with open(args.configs, 'r') as f:
        cf = json.load(f)['configs']
    pt.create_folder(args.save_path)

    height = cf['height']
    width = cf['width']
    classes_num = cf['n_class']

    model = EmbeddingCapsNet(num_classes=classes_num, ratio=cf['ratio'], cuda=torch.cuda.is_available())
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint)
    del checkpoint

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

    mAP, query_ids, gallery_ids = cal_mAP(model, query_loader, gallery_loader)

    _max = np.argmax(mAP)
    _min = np.argmin(mAP)

    q_max_img = query_loader.dataset[_max][0]
    q_min_img = query_loader.dataset[_min][0]

    g_max_img = list()
    for i in np.where(gallery_ids == query_ids[_max])[0]:
        g_max_img.append(gallery_loader.dataset[i][0])
    g_max_img = torch.stack(g_max_img, dim=0)

    g_min_img = list()
    for i in np.where(gallery_ids == query_ids[_min])[0]:
        g_min_img.append(gallery_loader.dataset[i][0])
    g_min_img = torch.stack(g_min_img, dim=0)

    save_img(model, q_max_img, q_min_img, g_max_img, g_min_img, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./Celeb-reID-light')
    parser.add_argument('--configs', type=str, default='./configs/Celeb-reID-light.json')
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default=None, help='If let None, GPU set automatically.')
    parser.add_argument('--save_path', type=str, default='./evaluate/outputs')
    args = parser.parse_args()

    # set gpu
    if args.gpu_id: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # modify relative path to absolute path
    args.data_root = os.path.abspath(args.data_root)

    visualize_feature(args)

