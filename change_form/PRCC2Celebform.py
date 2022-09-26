import glob
import shutil
import argparse
import ProcessingTools as pt


def main(prcc_path: str, save_path: str):
    """
    :param prcc_path:
    :param save_path:
    :return True:
    """

    pt.create_folder(f'{save_path}/train')
    pt.create_folder(f'{save_path}/gallery')
    pt.create_folder(f'{save_path}/query')

    # train
    train_id = sorted(glob.glob(f'{prcc_path}/rgb/train/*'))

    print('start train set')
    for id, fol in pt.ProgressBar(enumerate(train_id), finish_mark='finished train set'):
        val = fol.replace('train', 'val')
        imgs = sorted(glob.glob(f'{fol}/*.jpg') + glob.glob(f'{val}/*.jpg'))
        for n, img in enumerate(imgs):
            shutil.copy(img, f'{save_path}/train/{id + 1}_{n + 1}_0.jpg')

    ns = list()
    # test
    test_gallery_id = sorted(glob.glob(f'{prcc_path}/rgb/test/A/*'))
    print('start gallery set in test data')
    for id, fol in pt.ProgressBar(enumerate(test_gallery_id), finish_mark='finished gallery set in test data'):
        imgs = sorted(glob.glob(f'{fol}/*.jpg'))
        for n, img in enumerate(imgs):
            shutil.copy(img, f'{save_path}/gallery/{id + 1}_{n + 1}_1.jpg')
        ns.append(n + 1)

    test_query_id = sorted(glob.glob(f'{prcc_path}/rgb/test/C/*'))
    print('start gallery set in test data')
    for id, fol in pt.ProgressBar(enumerate(test_query_id), finish_mark='finished gallery set in test data'):
        imgs = sorted(glob.glob(f'{fol}/*.jpg'))
        for n, img in enumerate(imgs):
            shutil.copy(img, f'{save_path}/query/{id + 1}_{ns[id] + n + 1}_0.jpg')


    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prcc_root', type=str, default='./prcc')
    parser.add_argument('--save_path', type=str, default='./prcc_celeb_form')
    args = parser.parse_args()

    main(args.prcc_root, args.save_path)
