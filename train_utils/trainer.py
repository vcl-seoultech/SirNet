import torch
import ProcessingTools as pt
import loss.losses
import evaluate.pid_eval
import evaluate.visualize_recon


def adjust_lr(epoch, optimizer, lr, lr_decay):
    lr = lr * (0.1 ** (epoch // lr_decay))

    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

    return optimizer


def train(model, train_loader, all_loader, start, epoch, save_path, cf, logs, cuda: bool = False):
    model.train()

    base_param_ids = set(map(id, model.module.dense_feature.parameters())) if torch.cuda.device_count() > 1 else set(map(id, model.dense_feature.parameters()))

    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    if torch.cuda.device_count() > 1:
        param_groups = [
            {'params': model.module.dense_feature.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = [
            {'params': model.dense_feature.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.Adam(param_groups, lr=cf['lr'])

    tri_criterion = loss.losses.TripletLoss(cf['tri_margin'])
    sim_criterion = loss.losses.SimLoss(model, all_loader)
    capsule_criterion = torch.nn.CrossEntropyLoss()
    recon_criterion = loss.losses.DistillLoss(model.reconstructor, cf['ratio'], model.classifier_activation) if torch.cuda.device_count() <= 1 else loss.losses.DistillLoss(model.module.reconstructor, cf['ratio'], model.module.classifier_activation)
    if cuda:
        sim_criterion = sim_criterion.cuda()
        tri_criterion = tri_criterion.cuda()
        capsule_criterion = capsule_criterion.cuda()
        recon_criterion = recon_criterion.cuda()

    sim_criterion.update_c(cf['n_class'], cuda)

    for n in range(start, epoch):
        optimizer = adjust_lr(n, optimizer, cf['lr'], cf['lr_decay'])

        losses = {'lt': 0, 'l_id': 0, 'l_sim': 0, 'l_tri': 0, 'l_recon': 0}
        sim_criterion.update_c(cf['n_class'], cuda) if (n + 1) % cf['interval'] == (cf['warm up'] + 1) % cf['interval'] and (n + 1) > cf['warm up'] and cf['w_sim'] > 0 else None

        for i, inputs in pt.ProgressBar(enumerate(train_loader), finish_mark=None, max=len(train_loader)):
            optimizer.zero_grad()

            imgs, positive, (negative, pid_n), pids, _ = inputs

            if cuda:
                imgs = imgs.cuda()
                pids = pids.type(torch.LongTensor).cuda()
                pid_n = pid_n.type(torch.LongTensor).cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            caps_id, feature, recon = model(imgs)
            _, pos_outs, recon_p = model(positive)
            _, neg_outs, recon_n = model(negative)

            tri_loss = tri_criterion(feature, pos_outs, neg_outs) if cf['w_tri'] > 0 else 0
            sim_loss = sim_criterion(feature, pids, cuda) if (n + 1) > cf['warm up'] and cf['w_sim'] > 0 else 0
            id_loss = ((capsule_criterion(caps_id[0], pids) / cf['ratio'] * (cf['ratio'] - 1)) + (capsule_criterion(caps_id[1], pids)) / cf['ratio']) * 2
            id_loss_cam = capsule_criterion(caps_id[2], pids) if cf['w_recon'] > 0 else 0
            recon_loss = recon_criterion(recon, recon_p, recon_n, imgs, positive, pids, pid_n) if cf['w_recon'] > 0 else 0

            tri_loss = cf['w_tri'] * tri_loss
            sim_loss = cf['w_sim'] * sim_loss
            id_loss = cf['w_id'] * id_loss
            id_loss_cam = id_loss_cam
            recon_loss = cf['w_recon'] * recon_loss

            T_loss = tri_loss + sim_loss + id_loss + recon_loss + id_loss_cam
            losses['lt'] = losses['lt'] + float(T_loss) - float(id_loss_cam)
            losses['l_tri'] = losses['l_tri'] + float(tri_loss)
            losses['l_sim'] = losses['l_sim'] + float(sim_loss)
            losses['l_id'] = losses['l_id'] + float(id_loss)
            losses['l_recon'] = losses['l_recon'] + float(recon_loss)

            T_loss.backward()
            optimizer.step()

        print('\033[2K\r', end='')
        pt.print_write(f"[{n + 1}/{epoch}] "
                       f"Total Loss: {losses['lt'] / len(train_loader):0.5f}    "
                       f"triplet: {losses['l_tri'] / len(train_loader):0.3f}    "
                       f"id: {losses['l_id'] / len(train_loader):0.3f}    "
                       f"sim: {losses['l_sim'] / len(train_loader):0.5f}    "
                       f"recon: {losses['l_recon'] / len(train_loader):0.5f}", logs)

        # write logs
        logs.close()
        logs = open(logs.name, 'a')

        # save model
        if (n + 1) % cf['save epoch'] == 0:
            if torch.cuda.device_count() > 1: torch.save(model.module.state_dict(), f'{save_path}/model_{n + 1:04d}.pth.tar')
            else: torch.save(model.state_dict(), f'{save_path}/model_{n + 1:04d}.pth.tar')
            evaluate.pid_eval.pid_eval(model, all_loader, cf['ratio'], logs, cuda)
            evaluate.visualize_recon.image_recon(model, imgs, positive, negative, recon, recon_p, recon_n, cf, n, save_path)

    return model
