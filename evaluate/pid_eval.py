import torch
import ProcessingTools as pt


def pid_eval(model, data_loader, ratio, logs, cuda: bool = False):
    """
    evaluate pid accuracy (evaluate classification)
    :param model: pytorch model
    :param data_loader: data loader
    :param ratio: feature ratio
    :param logs: logs file
    :param cuda: using gpu or not
    :return: Ture
    """

    model.eval()
    print('Evaluating.')
    with torch.no_grad():
        num_correct = 0
        num_images = 0
        for inputs in pt.ProgressBar(data_loader, finish_mark=None):
            imgs, pids, _ = inputs
            if cuda:
                imgs = imgs.cuda()
                pids = pids.cuda()
            infer, _, _ = model(imgs)
            infer = (infer[0] / ratio * (ratio - 1)) + (infer[1] / ratio)
            infer = torch.argmax(infer, dim=1)

            num_correct = num_correct + torch.sum(infer == pids)
            num_images = num_images + imgs.shape[0]

        if cuda:
            num_correct = num_correct.cpu()

    print(f'\r{" " * 70}', end='')
    print(f'\rpid acc: {num_correct.item() * 100 / num_images:0.1f}%')
    print(f'pid acc: {num_correct.item() * 100 / num_images:0.1f}%', file=logs)
    model.train()

    return True
