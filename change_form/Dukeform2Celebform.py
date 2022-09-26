import glob
import shutil
import argparse
import ProcessingTools as pt


def main(duke_path: str, save_path: str):
    """
    :param duke_path:
    :param save_path:
    :return True:
    """

    pt.create_folder(f'{save_path}/train')
    pt.create_folder(f'{save_path}/gallery')
    pt.create_folder(f'{save_path}/query')

    # train
    train_imgs = pt.sorted_glob(f'{duke_path}/bounding_box_train/*.jpg')

    _id = 1
    print('start train set')
    for n, img in pt.ProgressBar(enumerate(train_imgs), finish_mark='finished train set'):
        img = img.replace('\\', '/')
        fore_id = int((img.split('/')[-1]).split('_')[0]) if n == 0 else fore_id
        if fore_id != int((img.split('/')[-1]).split('_')[0]):
            fore_id = int((img.split('/')[-1]).split('_')[0])
            _id = _id + 1

        cam = (img.split('/')[-1]).split('_')[1][1]
        shutil.copy(img, f'{save_path}/train/{_id}_{cam}_{n + 1}.jpg')

    # gallery
    print('start gallery set')
    gallery_id = sorted(glob.glob(f'{duke_path}/bounding_box_test/*.jpg'))
    for n, img in pt.ProgressBar(enumerate(gallery_id), finish_mark='finished gallery set in validation data'):
        img = img.replace('\\', '/')
        fore_id, cam = map(int, ((img.split('/')[-1]).split('_')[0], (img.split('/')[-1]).split('_')[1][1]))
        shutil.copy(img, f'{save_path}/gallery/{fore_id}_{cam}_{n + 1}.jpg')

    # query
    print('start query set')
    query_id = sorted(glob.glob(f'{duke_path}/query/*.jpg'))
    _id = 1
    for n, img in pt.ProgressBar(enumerate(query_id), finish_mark='finished gallery set in validation data'):
        img = img.replace('\\', '/')
        fore_id, cam = map(int, ((img.split('/')[-1]).split('_')[0], (img.split('/')[-1]).split('_')[1][1]))
        shutil.copy(img, f'{save_path}/query/{fore_id}_{cam}_{n + 1}.jpg')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duke_root', type=str, default='./duke')
    parser.add_argument('--save_path', type=str, default='./duke_celeb_form')
    args = parser.parse_args()

    main(args.duke_root, args.save_path)
