import shutil
import argparse
import ProcessingTools as pt


def main(vc_path: str, save_path: str):
    """
    :param vc_path:
    :param save_path:
    :return True:
    """

    pt.create_folder(f'{save_path}/train')
    pt.create_folder(f'{save_path}/gallery')
    pt.create_folder(f'{save_path}/query')

    # train
    train_imgs = pt.sorted_glob(f'{vc_path}/train/*.jpg')
    print('start train set')
    for n, img in pt.ProgressBar(enumerate(train_imgs), finish_mark='finished train set'):
        _id = int(img.split('/')[-1].split('-')[0])
        cam = int(img.split('/')[-1].split('-')[1])
        shutil.copy(img, f'{save_path}/train/{_id}_{cam}_{n + 1}.jpg')

    # test
    gallery_imgs = pt.sorted_glob(f'{vc_path}/gallery/*.jpg')
    print('start gallery set')
    for n, img in pt.ProgressBar(enumerate(gallery_imgs), finish_mark='finished gallery set'):
        _id = int(img.split('/')[-1].split('-')[0])
        cam = int(img.split('/')[-1].split('-')[1])
        shutil.copy(img, f'{save_path}/gallery/{_id}_{cam}_{n + 1}.jpg')

    query_imgs = pt.sorted_glob(f'{vc_path}/query/*.jpg')
    print('start query set')
    for n, img in pt.ProgressBar(enumerate(query_imgs), finish_mark='finished query set'):
        _id = int(img.split('/')[-1].split('-')[0])
        cam = int(img.split('/')[-1].split('-')[1])
        shutil.copy(img, f'{save_path}/query/{_id}_{cam}_{n + 1}.jpg')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vcc_root', type=str, default='./VC-Clothes')
    parser.add_argument('--save_path', type=str, default='./VC-Clothes_celeb_form')
    args = parser.parse_args()

    main(args.vcc_root, args.save_path)
