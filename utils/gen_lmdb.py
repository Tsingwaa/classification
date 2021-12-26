import argparse
import os

# import io
# import cv2
import lmdb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-lst-path",
                        type=str,
                        help="input list including image name and category")
    parser.add_argument("--map-lst-path",
                        type=str,
                        help="input list including image category and label")
    parser.add_argument("--img-dir", type=str, help="input image dir.")
    parser.add_argument("--lmdb-dir",
                        type=str,
                        help="directory to save data.mdb and lock.mdb file.")

    args = parser.parse_args()
    return args


def gen_lmdb(args):

    print("======> Start generating LMDB...")
    if not os.path.isdir(args.lmdb_dir):
        os.makedirs(args.lmdb_dir)

    lmdb_env = lmdb.open(args.lmdb_dir, map_size=8589934592 * 50)
    lmdb_txn = lmdb_env.begin(write=True)

    # Write map (label:ctg) into lmdb
    map_ctg2label = {}
    with open(args.map_lst_path, 'r') as f:
        for line in f.readlines():
            ctg, label = line.strip().split('\t')
            map_ctg2label[ctg] = label
            lmdb_txn.put(str(label).encode(), ctg.encode())

    with open(args.img_lst_path, 'r') as f:
        num_samples = 0
        for idx, line in tqdm(enumerate(f.readlines())):
            img_name, img_ctg = line.strip().split('\t')
            img_path = os.path.join(args.img_dir, img_name)
            img_label = map_ctg2label[img_ctg]

            if not os.path.exists(img_path):  # filter
                continue

            with open(img_path, 'rb') as f:
                # 'rb' ensures that f.read() is 'Byte'
                img_buffer = f.read()

            img_key = 'image-%09d' % (num_samples + 1)
            label_key = 'label-%09d' % (num_samples + 1)

            # Ensure that key and value are both 'Byte'
            lmdb_txn.put(img_key.encode(), img_buffer, overwrite=False)
            lmdb_txn.put(label_key.encode(),
                         str(label).encode(),
                         overwrite=False)

            num_samples += 1

        # Write total sample number into lmdb
        lmdb_txn.put('num-samples'.encode(), str(num_samples).encode())
        lmdb_txn.commit()

    lmdb_env.sync()
    lmdb_env.close()
    print(f"=====> Done. \nLMDB save path: '{args.lmdb_dir}'")
    print("===========================================================")


if __name__ == "__main__":
    args = parse_args()
    gen_lmdb(args)
