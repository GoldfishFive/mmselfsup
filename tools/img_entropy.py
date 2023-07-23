'/home/data1/group1/datasets/COCO2017/COCO_2017/raw/Images/train2017'
import multiprocessing
import os
import numpy as np
from skimage import data
from skimage.measure import shannon_entropy
import cv2
import matplotlib.pyplot as plt

def get_entropy(img_path):
    save_root = '/home/data1/group1/datasets/COCO2017/COCO_2017/raw/Images/visual'
    img_name = os.path.basename(img_path)
    # ca = data.camera()
    img = cv2.imread(img_path)

    en = shannon_entropy(img)
    # print(en)

    cv2.putText(img, str(en), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(save_root,img_name), img)
    # cv2.imshow('img',ca)
    # cv2.waitKey(0)

def get_COCO_Train_entropy():
    pass

def multi_in1kval_entropy(img_path):
    save_root = '/home/data1/group1/datasets/imagenet_val_FH_500_500/'
    img = cv2.imread(img_path)
    en = shannon_entropy(img)
    c = os.path.basename(os.path.dirname(img_path)) # class name
    i = os.path.basename(img_path)
    npy_path = os.path.join(save_root, c, i.replace('JPEG', 'npy'))
    save_data = np.load(npy_path,allow_pickle=True).item()
    save_data['entropy'] = en
    np.save(npy_path, save_data)

def get_imagenet1k_val_entropy():
    dataset_root = '/home/user01/datasets/imagenet_val/'
    save_root = '/home/data1/group1/datasets/imagenet_val_FH_500_500/'

    img_list = []
    for c in os.listdir(dataset_root):
        c_root = os.path.join(dataset_root, c)
        os.makedirs(os.path.join(save_root, c), exist_ok=True)
        for i in os.listdir(c_root):
            img_list.append(os.path.join(c_root, i))
    print(img_list)
    pool = multiprocessing.Pool(processes=10)
    pool.map(multi_in1kval_entropy, img_list)

def analyse_imagenet1k_val_entropy():
    save_root = '/home/data1/group1/datasets/imagenet_val_FH_500_500/'

    entropy_list = []
    for c in os.listdir(save_root):
        c_root = os.path.join(save_root, c)
        for i in os.listdir(c_root):
            npy_path=os.path.join(c_root, i)
            save_data = np.load(npy_path, allow_pickle=True).item()
            entropy_list .append(save_data['entropy'])
    print(entropy_list)
    # entropy_list = [for e in entropy_list]
    plt.hist(entropy_list, bins=100)
    plt.show()
    print('max entropy of in1kval is: ', max(entropy_list))
    print('min entropy of in1kval is: ', min(entropy_list))
    print('mean entropy of in1kval is: ', np.mean(entropy_list))


if __name__ == "__main__":
    # get_imagenet1k_val_entropy()
    analyse_imagenet1k_val_entropy()
    exit()
    img_root = '/home/data1/group1/datasets/COCO2017/COCO_2017/raw/Images/train2017'
    count = 0
    for i in os.listdir(img_root):
        count += 1
        if count == 50:
            break
        get_entropy(os.path.join(img_root,i))