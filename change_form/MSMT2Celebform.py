import shutil
import argparse
import ProcessingTools as pt
import os.path


def main(msmt_path: str, save_path: str):
    """
    :param msmt_path:
    :param save_path:
    :return True:
    """

    pt.create_folder(f'{save_path}/train')
    pt.create_folder(f'{save_path}/gallery')
    pt.create_folder(f'{save_path}/query')

    # train
    print('start train set')

    for _id in pt.ProgressBar(range(1041), finish_mark='finished train set'):
        train_imgs = pt.sorted_glob(f'{msmt_path}/mask_train_v2/{_id:04d}/*.jpg')

        for img in train_imgs:
            num_cam = os.path.basename(img).split('_')
            n = num_cam[1]
            cam = num_cam[2]

            shutil.copy(img, f'{save_path}/train/{_id + 1}_{cam}_{n}.jpg')

    # gallery
    print('start gallery set')
    with open(f'{msmt_path}/list_gallery.txt', 'r') as f:
        lines = f.readlines()

        for line in pt.ProgressBar(lines, finish_mark='finished gallery set in validation data'):
            img = line.split(' ')[0]
            id_num_cam = img.split('/')[1].split('_')
            _id, n, cam = id_num_cam[0], id_num_cam[1], id_num_cam[2]
            shutil.copy(f'{msmt_path}/mask_test_v2/{img}', f'{save_path}/gallery/{_id}_{cam}_{n}.jpg')

    # query
    print('start query set')
    with open(f'{msmt_path}/list_query.txt', 'r') as f:
        lines = f.readlines()

        for line in pt.ProgressBar(lines, finish_mark='finished query set in validation data'):
            img = line.split(' ')[0]
            id_num_cam = img.split('/')[1].split('_')
            _id, n, cam = id_num_cam[0], id_num_cam[1], id_num_cam[2]
            shutil.copy(f'{msmt_path}/mask_test_v2/{img}', f'{save_path}/query/{_id}_{cam}_{n}.jpg')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--msmt_root', type=str, default='./msmt')
    parser.add_argument('--save_path', type=str, default='./msmt_celeb_form')
    args = parser.parse_args()

    main(args.msmt_root, args.save_path)
