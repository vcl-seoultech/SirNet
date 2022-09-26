import torch
import numpy as np
import torchvision
import ProcessingTools as pt
import torch.nn.functional


class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    """

    def __init__(self, tri_margin):
        """
        Initialize triplet loss
        :param tri_margin: margin of triplet loss
        """

        super().__init__()
        self.tri_margin = tri_margin

    def forward(self, feature, pos_feature, neg_feature):
        """
        Calculate triplet loss
        :param feature: query feature
        :param pos_feature: positive feature
        :param neg_feature: negative feature
        :return: triplet loss (torch.sum(torch.maximum(pos_dis - neg_dis + tri_margin, torch.zeros_like(pos_dis))))
        """

        pos_dis = torch.sqrt(torch.sum((feature - pos_feature) ** 2, dim=1))
        neg_dis = torch.sqrt(torch.sum((feature - neg_feature) ** 2, dim=1))

        return torch.sum(torch.maximum(pos_dis - neg_dis + self.tri_margin, torch.zeros_like(pos_dis))) / feature.shape[0]


class SimLoss(torch.nn.Module):
    def __init__(self, model, data_loader):
        """
        Sample independent maximum discrepancy loss
        :param model:
        :param data_loader: data loader for obtain center points
        """

        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.warmup = False

        # initialize centers
        self.centers = 0

    def forward(self, features, pids, cuda: bool = False):
        """
        Calculate Sim loss.
        :param features: input feature
        :param cuda: using gpu or not
        :return: sim loss
        """

        loss = 0
        if not self.warmup: return loss

        centers = torch.stack((self.centers, ) * features.shape[0], dim=0)
        features = torch.stack((features, ) * centers.shape[1], dim=1)  # (batch size, the number of classes, dimension)
        dis_n = features.shape[2] - torch.sum((features - centers) ** 2, dim=2)  # (batch size, the number of classes)
        softmax = torch.softmax(dis_n, dim=1)
        pids_where = torch.eye(centers.shape[1], dtype=int).cuda().index_select(dim=0, index=pids).cuda() if cuda else torch.eye(centers.shape[1], dtype=int).index_select(dim=0, index=pids)
        pids_where = torch.autograd.Variable(pids_where)
        logit = torch.sum(softmax * pids_where, dim=1)

        return torch.sum(-torch.log(logit)) / features.shape[0]

    def update_c(self, num_class, cuda: bool = False):
        """
        Update center points.
        :param num_class: the number of class
        :param cuda: using gpu or not
        :return: True
        """

        print('Update classes center points')
        self.model.eval()
        with torch.no_grad():
            # initialize
            all_feature = list()
            all_pids = list()
            for images, pids, _ in pt.ProgressBar(self.data_loader, finish_mark=None, max=len(self.data_loader)):
                if cuda:
                    images = images.cuda()
                    pids = pids.cuda()
                _, feature, _ = self.model(images)
                feature = feature.detach().cpu().numpy()
                pids = pids.detach().cpu().numpy()

                all_feature.append(feature)
                all_pids.append(pids)

            all_feature = np.concatenate(all_feature, axis=0)
            all_pids = np.concatenate(all_pids, axis=0)

            self.model.train()

            centers = [0 for _ in range(num_class)]
            for idx in range(num_class):
                class_feature = all_feature[all_pids == idx]
                centers[idx] = np.sum(class_feature, axis=0) / class_feature.shape[0]

            self.centers = torch.tensor(centers).cuda()
            self.warmup = True

        print('\r\033[2KDone.')
        return True


class ReconstructionLoss(torch.nn.Module):
    """
    Image reconstruction loss.
    """
    def __init__(self, norm: int = 1, size=(64, 64)):
        """
        Initialize reconstruction loss.
        :param norm: norm for reconstruction loss
        """
        super().__init__()
        self.norm = norm
        self.size = size

    def forward(self, feature_recon, gt_images):
        """
        Calculate reconstruction loss from digit caps and ground truth images.
        :param feature_recon: Reconstructed images
        :param gt_images: Ground truth images
        :return: Reconstruction loss
        """

        recon = feature_recon[0]
        attention = torch.reshape(torch.nn.functional.interpolate(torch.sum(feature_recon[1], dim=1, keepdim=True) / feature_recon[1].shape[1], self.size, mode='bilinear', align_corners=False), (gt_images.shape[0], -1))
        zeros = torch.zeros_like(recon, requires_grad=False)
        zeros[attention >= torch.mean(attention)] = 1
        n = torch.sum(zeros)
        if n == 0: return 0

        gt_images = torch.nn.functional.interpolate(gt_images, self.size, mode='bilinear', align_corners=False)
        gray_gt = torch.reshape(torchvision.transforms.Grayscale()(gt_images), (gt_images.shape[0], -1))

        diff = (recon - gray_gt) * zeros
        loss = torch.sum(diff ** self.norm) ** (1 / self.norm) if self.norm % 2 == 0 else torch.sum(
            torch.abs(diff) ** self.norm) ** (1 / self.norm)
        return loss / n


class DistillLoss(torch.nn.Module):
    """
    Distillation loss.
    """
    def __init__(self, reconstructor, ratio, weight, size=(64, 64)):
        """
        Initialize distillation loss.
        :param reconstructor: reconstructor of model
        """
        super().__init__()
        self.reconstructor = reconstructor
        self.size = size
        self.ratio = ratio
        self.weight = weight

    def mix_loss(self, recon, feature_fore, feature_back, pid_fore, pid_back):
        attention_fore = self.weight.weight[pid_fore][..., None, None] * feature_fore
        attention_back = self.weight.weight[pid_back][..., None, None] * feature_back

        sum_fore = torch.reshape(torch.sum(attention_fore, dim=1), (attention_fore.shape[0], -1))
        sum_back = torch.reshape(torch.sum(attention_back, dim=1), (attention_fore.shape[0], -1))

        mean_fore = torch.mean(sum_fore, dim=1)[..., None]
        mean_back = torch.mean(sum_back, dim=1)[..., None]

        fore_mask = torch.zeros_like(sum_fore)
        fore_mask[sum_fore > mean_fore] = 1
        back_mask = torch.zeros_like(sum_fore)
        back_mask[sum_back < mean_back] = 1
        back_mask[sum_fore > mean_fore] = 0

        shape = list(attention_fore.shape)
        shape[1] = 1
        fore_mask = torch.reshape(fore_mask, shape)
        back_mask = torch.reshape(back_mask, shape)

        gt = torch.reshape(feature_fore * fore_mask + feature_back * back_mask, (attention_fore.shape[0], -1))
        mask = torch.reshape(torch.cat((fore_mask + back_mask, ) * attention_fore.shape[1]), (attention_fore.shape[0], -1))

        return torch.sum(torch.abs(gt - recon) * mask) / torch.sum(mask) * 2

    def img_reshape(self, imgs):
        imgs = torch.nn.functional.interpolate(imgs, self.size, mode='bilinear', align_corners=False)
        imgs = torch.reshape(torchvision.transforms.Grayscale()(imgs), (imgs.shape[0], -1))

        return imgs

    def forward(self, out, pos_out, neg_out, imgs, pos_imgs, pid, pid_n):
        """
        Calculate distillation loss from digit caps and ground truth images.
        :param out: Reconstructed images
        :param pos_out: Ground truth images
        :param neg_out: Ground truth images
        :return: Reconstruction loss
        :param imgs: anchor images
        :param pos_imgs: positive images
        :param pid: pid
        :param pid_n: negative pid
        :return: distillation loss
        """

        feature_n = out[0].shape[1] // self.ratio * (self.ratio - 1)
        _id = out[0][:, :feature_n]
        back = out[0][:, feature_n:]
        id_p = pos_out[0][:, :feature_n]
        back_p = pos_out[0][:, feature_n:]
        id_n = neg_out[0][:, :feature_n]
        back_n = neg_out[0][:, feature_n:]

        imgs = self.img_reshape(imgs)
        pos_imgs = self.img_reshape(pos_imgs)

        recon, _ = self.reconstructor(out[0])

        pos_mix0, _ = self.reconstructor(torch.cat((_id, back_p), dim=1))
        pos_mix1, _ = self.reconstructor(torch.cat((id_p, back), dim=1))

        _, neg_mix0 = self.reconstructor(torch.cat((_id, back_n), dim=1))
        _, neg_mix1 = self.reconstructor(torch.cat((id_n, back), dim=1))

        recon_loss = torch.mean(torch.abs(recon - imgs))
        pos_loss0 = torch.mean(torch.abs(pos_mix0 - pos_imgs))
        pos_loss1 = torch.mean(torch.abs(pos_mix1 - imgs))
        neg_loss0 = self.mix_loss(neg_mix0, out[1], neg_out[1], pid, pid_n)
        neg_loss1 = self.mix_loss(neg_mix1, neg_out[1], out[1], pid_n, pid)

        return (recon_loss + pos_loss0 + pos_loss1 + neg_loss0 + neg_loss1) / 5
