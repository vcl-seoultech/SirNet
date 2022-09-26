import glob
import shutil
import argparse
import ProcessingTools as pt
import json


def main(cuhk03_path: str, save_path: str):
    """
    :param cuhk03_path:
    :param save_path:
    :return True:
    """

    with open(f'{cuhk03_path}/splits_classic_detected.json', 'r') as f:
        j_file = json.load(f)

    pt.create_folder(f'{save_path}/train')
    pt.create_folder(f'{save_path}/gallery')
    pt.create_folder(f'{save_path}/query')

    # train
    train_imgs = j_file[0]['train']

    ids = list()
    print('start train set')
    for n, img in pt.ProgressBar(enumerate(train_imgs), finish_mark='finished train set'):
        img = img[0].replace('\\', '/').split('/')[-1]
        fore_id, cam = img[:5], int(img[6])
        if fore_id not in ids:
            ids.append(fore_id)
        _id = ids.index(fore_id) + 1
        shutil.copy(f'{cuhk03_path}/images_detected/{img}', f'{save_path}/train/{_id}_{cam}_{n + 1}.png')

    ids = list()
    # gallery
    print('start gallery set')
    gallery_id = j_file[0]['gallery']
    for n, img in pt.ProgressBar(enumerate(gallery_id), finish_mark='finished gallery set in validation data'):
        img = img[0].replace('\\', '/').split('/')[-1]
        fore_id, cam = img[:5], int(img[6])
        if fore_id not in ids:
            ids.append(fore_id)
        _id = ids.index(fore_id) + 1
        shutil.copy(f'{cuhk03_path}/images_detected/{img}', f'{save_path}/gallery/{_id}_{cam}_{n + 1}.png')

    # query
    print('start query set')
    query_id = j_file[0]['query']
    for n, img in pt.ProgressBar(enumerate(query_id), finish_mark='finished gallery set in validation data'):
        img = img[0].replace('\\', '/').split('/')[-1]
        fore_id, cam = img[:5], int(img[6])
        _id = ids.index(fore_id) + 1
        shutil.copy(f'{cuhk03_path}/images_detected/{img}', f'{save_path}/query/{_id}_{cam}_{n + 1}.png')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuhk03_root', type=str, default='./cuhk03')
    parser.add_argument('--save_path', type=str, default='./cuhk03_celeb_form')
    args = parser.parse_args()

    main(args.cuhk03_root, args.save_path)
