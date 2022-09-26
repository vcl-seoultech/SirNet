import glob
import shutil
import argparse
import ProcessingTools as pt


def main(last_path: str, save_path: str):
    """
    :param last_path:
    :param save_path:
    :return True:
    """

    pt.create_folder(f'{save_path}/train')
    pt.create_folder(f'{save_path}/val/gallery')
    pt.create_folder(f'{save_path}/val/query')
    pt.create_folder(f'{save_path}/test/gallery')
    pt.create_folder(f'{save_path}/test/query')

    # train
    train_id = sorted(glob.glob(f'{last_path}/train/*'))

    print('start train set')
    for id, fol in pt.ProgressBar(enumerate(train_id), finish_mark='finished train set'):
        imgs = sorted(glob.glob(f'{fol}/*.jpg'))
        for n, img in enumerate(imgs):
            shutil.copy(img, f'{save_path}/train/{id + 1}_{n + 1}_0.jpg')

    # val
    print('start gallery set in validation data')
    val_gallery_id = sorted(glob.glob(f'{last_path}/val/gallery/*'))[1:]    # get rid of 000000
    for id, fol in pt.ProgressBar(enumerate(val_gallery_id), finish_mark='finished gallery set in validation data'):
        imgs = sorted(glob.glob(f'{fol}/*.jpg'))
        for n, img in enumerate(imgs):
            shutil.copy(img, f'{save_path}/val/gallery/{id + 1}_0_0.jpg')

    imgs = sorted(glob.glob(f'{last_path}/val/query/*.jpg'))

    ids = list()
    id = 0
    be_pid = 0
    for img in imgs:
        pid = img.split('/')[-1].split('_')[0]
        if be_pid != pid: id = id + 1
        ids.append(id)
        be_pid = pid

    print('start query set in validation data')
    for n, img in pt.ProgressBar(enumerate(imgs), finish_mark='finished query set in validation data'):
        shutil.copy(img, f'{save_path}/val/query/{ids[n]}_1_{n}.jpg')

    # test
    test_gallery_id = sorted(glob.glob(f'{last_path}/test/gallery/*'))[1:]  # get rid of 000000
    print('start gallery set in test data')
    for id, fol in pt.ProgressBar(enumerate(test_gallery_id), finish_mark='finished gallery set in test data'):
        imgs = sorted(glob.glob(f'{fol}/*.jpg'))
        for n, img in enumerate(imgs):
            shutil.copy(img, f'{save_path}/test/gallery/{id + 1}_2_0.jpg')

    imgs = sorted(glob.glob(f'{last_path}/test/query/*.jpg'))

    ids = list()
    id = 0
    be_pid = 0
    for img in imgs:
        pid = img.split('/')[-1].split('_')[0]
        if be_pid != pid: id = id + 1
        ids.append(id)
        be_pid = pid

    print('start query set in test data')
    for n, img in pt.ProgressBar(enumerate(imgs), finish_mark='finished query set in test data'):
        shutil.copy(img, f'{save_path}/test/query/{ids[n]}_3_{n}.jpg')


    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--last_root', type=str, default='./last')
    parser.add_argument('--save_path', type=str, default='./last_celeb_form')
    args = parser.parse_args()

    main(args.last_root, args.save_path)
