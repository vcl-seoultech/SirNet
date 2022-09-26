import glob
import shutil
import argparse
import ProcessingTools as pt


def main(ltcc_path: str, save_path: str):
    """
    :param ltcc_path:
    :param save_path:
    :return True:
    """

    pt.create_folder(f'{save_path}/train')
    pt.create_folder(f'{save_path}/gallery')
    pt.create_folder(f'{save_path}/query')

    # train
    train_imgs = sorted(glob.glob(f'{ltcc_path}/train/*'))

    _id = 1
    print('start train set')
    for n, img in pt.ProgressBar(enumerate(train_imgs), finish_mark='finished train set'):
        fore_id = int((img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1 if n == 0 else fore_id
        if fore_id != int((img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1:
            fore_id = int((img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1
            _id = _id + 1

        cam = (img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[2][1:]
        shutil.copy(img, f'{save_path}/train/{_id}_{cam}_{n + 1}.png')

    # gallery
    print('start gallery set')
    gallery_id = sorted(glob.glob(f'{ltcc_path}/test/*'))
    _id = 1
    for n, img in pt.ProgressBar(enumerate(gallery_id), finish_mark='finished gallery set in validation data'):
        fore_id = int(
            (img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1 if n == 0 else fore_id
        if fore_id != int((img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1:
            fore_id = int((img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1
            _id = _id + 1

        cam = (img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[2][1:]
        shutil.copy(img, f'{save_path}/gallery/{_id}_{cam}_{n + 1}.png')

    # query
    print('start query set')
    query_id = sorted(glob.glob(f'{ltcc_path}/query/*'))
    _id = 1
    for n, img in pt.ProgressBar(enumerate(query_id), finish_mark='finished gallery set in validation data'):
        fore_id = int(
            (img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1 if n == 0 else fore_id
        if fore_id != int((img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1:
            fore_id = int((img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[0]) + 1
            _id = _id + 1

        cam = (img.split('\\')[-1] if '\\' in img else img.split('/')[-1]).split('_')[2][1:]
        shutil.copy(img, f'{save_path}/query/{_id}_{cam}_{n + 1}.png')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ltcc_root', type=str, default='./ltcc')
    parser.add_argument('--save_path', type=str, default='./ltcc_celeb_form')
    args = parser.parse_args()

    main(args.ltcc_root, args.save_path)
