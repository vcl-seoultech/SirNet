from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter

from .utils import to_torch
import models.ECNet
import torchvision
import numpy as np


def extract_cnn_feature(model, inputs):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs)
        #inputs = Variable(inputs, volatile=True)

        # _, _, (outputs, _) = model(inputs)
        # outputs = models.ECNet.l2(outputs)
        #
        # inputs_f = torchvision.transforms.RandomHorizontalFlip(p=1)(inputs)
        # _, _, (outputs_f, _) = model(inputs_f)
        # outputs = (models.ECNet.l2(outputs_f) + outputs) / 2

        _, _, (outputs, feature) = model(inputs)
        mean_pool = torch.reshape(torch.nn.AvgPool2d(feature.shape[2:4])(feature), feature.shape[0:2])
        outputs = torch.cat((models.ECNet.l2(outputs), mean_pool * 0.55), dim=1)

        inputs_f = torchvision.transforms.RandomHorizontalFlip(p=1)(inputs)
        _, _, (outputs_f, feature_f) = model(inputs_f)
        mean_pool_f = torch.reshape(torch.nn.AvgPool2d(feature_f.shape[2:4])(feature_f), feature_f.shape[0:2])
        outputs = (torch.cat((models.ECNet.l2(outputs_f), mean_pool_f * 0.55), dim=1) + outputs) / 2

        outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=1):
    model.eval()
    with torch.no_grad():
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        labels = OrderedDict()

        end = time.time()
        for i, (img, pids, name) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, img.cuda())
            for fname, output, pid in zip(name, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    print(x.size())
    print(y.size())
    x = x.view(m, -1)
    y = y.view(n, -1)
    print(x.size())
    print(y.size())
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    #dist.addmm_(1, -2, x, y.t())
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist


def evaluate_all(distmat, logs, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('\nMean AP: {:4.1%}'.format(mAP), file=logs)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'celeb': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores', file=logs)
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['celeb'][k - 1]), file=logs)
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['celeb'][k - 1]))

    return cmc_scores['celeb'][0]


class Evaluator(object):
    def __init__(self, model, rerank:bool = True):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, logs, rerank: bool = False, k1=20, k2=6, lambda_value=0.1):
        query_features, _ = extract_features(self.model, query_loader, 1)
        gallery_features, _ = extract_features(self.model, gallery_loader, 1)

        if rerank:
            query_features = torch.stack([query_features[i] for i in query_features.keys()], dim=0)
            gallery_features = torch.stack([gallery_features[i] for i in gallery_features.keys()], dim=0)
            distmat = re_ranking(query_features, gallery_features, k1=k1, k2=k2, lambda_value=lambda_value)
        else:
            distmat = pairwise_distance(query_features, gallery_features, query, gallery)

        return evaluate_all(distmat, logs, query=query, gallery=gallery), distmat


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        # distmat.addmm_(1,-2,feat,feat.t())
        distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist