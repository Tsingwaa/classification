
def split_dataset(data_root, dataset_name, split_ratio=0.8, manual_seed=0):
    """Split whole dataset into train and test dataset at ratio of 4:1.
    Args:
        data_root: save all dataset
        dataset_name: Dataset to be split
        split_ratio: Train samples account for split_ratio of all samples.
        manual_seed: seed for shuffle
    """

    import os
    import random
    import shutil
    from os.path import join, isdir
    from tqdm import tqdm

    random.seed(manual_seed)

    dataset_root = join(data_root, dataset_name)
    train_root = join(data_root, dataset_name + '_train')
    test_root = join(data_root, dataset_name + '_test')

    for class_name in sorted(os.listdir(dataset_root)):
        class_dir = join(dataset_root, class_name)
        if not isdir(class_dir):
            continue

        img_fnames = os.listdir(class_dir)
        random.shuffle(img_fnames)
        train_num = int(0.8 * len(img_fnames))

        # new class directory for trainset and testset
        train_dir = join(train_root, class_name)
        os.makedirs(train_dir, exist_ok=True)
        test_dir = join(test_root, class_name)
        os.makedirs(test_dir, exist_ok=True)

        print(f"\nProcessing '{class_name}'...")
        for i, img_fname in tqdm(enumerate(img_fnames)):
            src_fpath = join(class_dir, img_fname)
            if i < train_num:
                dst_fpath = join(train_dir, img_fname)
            else:
                dst_fpath = join(test_dir, img_fname)

            shutil.copyfile(src_fpath, dst_fpath)

    print("===> All is done.")


if __name__ == '__main__':
    data_root = "/home/waa/Data/Caltech"
    dataset_name = "5classesfrom256"
    split_ratio = 0.8

    split_dataset(data_root, dataset_name, split_ratio)
