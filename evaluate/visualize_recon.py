import torch
import argparse
import os
import json
import ProcessingTools as pt
from models.ECNet import EmbeddingCapsNet
import numpy as np
import cv2
import ProcessingTools.torch_tools as tt
from PIL import Image
from torchvision import transforms as T
import torchvision
import torch.nn.functional


def image_recon(model, imgs, positive, negative, recon, recon_p, recon_n, cf, n, save_path):
    """
    image reconstruction
    :param model: model
    :param imgs: input images
    :param positive: positive images
    :param negative: negative images
    :param recon: reconstruction
    :param recon_p: reconstruction positive
    :param recon_n: reconstruction negative
    :param cf: configuration
    :param n: present epoch
    :param save_path: save path
    :return: True
    """

    model.eval()
    reconstructor = model.reconstructor if torch.cuda.device_count() <= 1 else model.module.reconstructor
    feature_n = recon[0].shape[1] // cf['ratio'] * (cf['ratio'] - 1)
    with torch.no_grad():
        img = torch.unsqueeze(torchvision.transforms.Grayscale()(imgs[0]), dim=1)
        pos = torch.unsqueeze(torchvision.transforms.Grayscale()(positive[0]), dim=1)
        neg = torch.unsqueeze(torchvision.transforms.Grayscale()(negative[0]), dim=1)
        img_t = torch.cat((img, pos, neg), dim=-1)
        tt.torch_imgs_save(img_t, f'{save_path}/recon/gt{n + 1}_')

        gray, _ = reconstructor(recon[0][0][None, ...])
        pos_mix0, _ = reconstructor(
            torch.cat((recon[0][0][:feature_n][None, ...], recon_p[0][0][feature_n:][None, ...]), dim=1))
        pos_mix1, _ = reconstructor(
            torch.cat((recon_p[0][0][:feature_n][None, ...], recon[0][0][feature_n:][None, ...]), dim=1))
        neg_mix0, _ = reconstructor(
            torch.cat((recon[0][0][:feature_n][None, ...], recon_n[0][0][feature_n:][None, ...]), dim=1))
        neg_mix1, _ = reconstructor(
            torch.cat((recon_n[0][0][:feature_n][None, ...], recon[0][0][feature_n:][None, ...]), dim=1))

        w = round(gray.shape[1] ** 0.5)
        gray = torch.reshape(gray, (-1, 1, w, w))
        gray = torch.nn.functional.interpolate(gray, img.shape[2:4], mode='bilinear', align_corners=False)

        pos_mix0 = torch.reshape(pos_mix0, (-1, 1, w, w))
        pos_mix0 = torch.nn.functional.interpolate(pos_mix0, img.shape[2:4], mode='bilinear', align_corners=False)
        pos_mix1 = torch.reshape(pos_mix1, (-1, 1, w, w))
        pos_mix1 = torch.nn.functional.interpolate(pos_mix1, img.shape[2:4], mode='bilinear', align_corners=False)

        neg_mix0 = torch.reshape(neg_mix0, (-1, 1, w, w))
        neg_mix0 = torch.nn.functional.interpolate(neg_mix0, img.shape[2:4], mode='bilinear', align_corners=False)
        neg_mix1 = torch.reshape(neg_mix1, (-1, 1, w, w))
        neg_mix1 = torch.nn.functional.interpolate(neg_mix1, img.shape[2:4], mode='bilinear', align_corners=False)

        stacked_pos = torch.cat((gray, pos_mix0, pos_mix1), dim=3)
        stacked_neg = torch.cat((torch.zeros_like(gray), neg_mix0, neg_mix1), dim=3)
        tt.torch_imgs_save(torch.cat((stacked_pos, stacked_neg), dim=2), f'{save_path}/recon/recon{n + 1}_')

    model.train()
    return True


def recon_one(save_path: str, i, model, img, pid=None, img_mix=None, ratio=None):
    """
    return a reconstructed images and feature map
    :param save_path: image save path
    :param i: image number
    :param model: input network
    :param img: a normalized image
    :param pid: a image class
    :param img_mix: a back ground image to mix with imgs
    :param ratio: ratio of id features
    :return:
    """

    model.eval()
    with torch.no_grad():
        id, _, (features, feature_map) = model(img)
        pid = torch.argmax(id[0], dim=1) if pid is None else pid
        cam = model.classifier_activation.weight[pid][..., None, None] * feature_map
        cam = torch.nn.functional.interpolate(torch.sum(cam.cpu(), dim=1, keepdim=True), img.shape[2:], mode='bilinear', align_corners=False)
        imgs = cv2.cvtColor(np.array(tt.torch_img_denormalize(img.cpu())[0].permute(1, 2, 0)).astype('uint8'), cv2.COLOR_BGR2GRAY)[..., None]
        cam = cv2.applyColorMap(np.array(tt.torch_img_denormalize(cam)[0].permute(1, 2, 0).detach()).astype('uint8'), cv2.COLORMAP_JET)
        cv2.imwrite(f'{save_path}/cam{i:04}.png', (imgs[:, :, ::-1] / 2 + cam / 2))

        if img_mix is not None:
            feature_n = features.shape[1] // ratio * (ratio - 1)
            reconstructor = model.reconstructor if torch.cuda.device_count() <= 1 else model.module.reconstructor
            _, _, (features_mix, _) = model(img_mix)
            mix, _ = reconstructor(torch.cat((features[:, :feature_n], features_mix[:, feature_n:]), dim=1))
            t = round(np.sqrt(mix.shape[1]))
            mix = torch.reshape(mix, (1, 1, t, t))
            mix = torch.nn.functional.interpolate(mix, img.shape[2:], mode='bilinear', align_corners=False)
            tt.torch_imgs_save(mix.cpu(), f'{save_path}/recon{i:04}')

    model.train()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', type=str, default='')
    parser.add_argument('--back_img', type=str, default=None)
    parser.add_argument('--configs', type=str, default='./configs/Celeb-reID-light.json')
    parser.add_argument('--model', type=str)
    parser.add_argument('--gpu_id', type=str, default=None, help='If let None, GPU set automatically.')
    parser.add_argument('--save_path', type=str, default='./evaluate/outputs')
    parser.add_argument('--pid', type=bool, default=False)
    args = parser.parse_args()

    # set gpu
    if args.gpu_id: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # modify relative path to absolute path
    args.data_root = os.path.abspath(args.imgs_dir)
    args.back_img = os.path.abspath(args.back_img) if args.back_img is not None else None

    with open(args.configs, 'r') as f:
        cf = json.load(f)['configs']
    pt.create_folder(f'{args.save_path}')

    height = cf['height']
    width = cf['width']
    classes_num = cf['n_class']

    # load model
    model = EmbeddingCapsNet(num_classes=classes_num, ratio=cf['ratio'], cuda=torch.cuda.is_available())
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint)
    model.eval()

    # check GPU env
    if torch.cuda.device_count() > 1:
        print(f'{torch.cuda.device_count()} GPUs are detected.')
        model = torch.nn.DataParallel(model.cuda())
    elif torch.cuda.is_available():
        print('1 GPU is detected.')
        model = model.cuda()
    else:
        print('run on cpu.')
    cuda = torch.cuda.is_available()

    imgs = pt.sorted_glob(f'{args.imgs_dir}/*.jpg') + pt.sorted_glob(f'{args.imgs_dir}/*.png')
    transform = T.Compose([
        T.Resize(size=(height, width), interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    back_img = torch.unsqueeze(transform(Image.open(args.back_img)), dim=0) if args.back_img is not None else None

    for i, img in pt.ProgressBar(enumerate(imgs), finish_mark=None, max=len(imgs)):
        img = transform(Image.open(img))

        if cuda:
            img = img.cuda()
            back_img = back_img.cuda() if args.back_img is not None else None

        model.eval()
        img = torch.unsqueeze(img, dim=0)
        recon_one(args.save_path, i, model, img, pid=None, img_mix=back_img, ratio=cf['ratio'])
