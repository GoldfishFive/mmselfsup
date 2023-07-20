import os

import cv2
import numpy as np
import random
from skimage.segmentation import felzenszwalb

def selective_search(img, h, w, res_size=(1280,760)):
    img_det = np.array(img)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if res_size is not None:
        img_det = cv2.resize(img_det, res_size)

    ss.setBaseImage(img_det)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process().astype('float32')
    print(boxes)
    if res_size is not None:
        boxes /= res_size[0]
        boxes *= np.array([w, h, w, h])

    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes

def get_rect(img):
    # read image
    im = cv2.imread(img)
    # resize image
    newHeight = 200
    newWidth = int(im.shape[1] * 200 / im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # ss.switchToSelectiveSearchFast()

    ss.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 50

    while True:
        # create a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Output", imOut);

        # record key press
        k = cv2.waitKey(0) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()



if __name__ == '__main__':
    dataset_root = '/home/user01/datasets/imagenet_val/'
    save_root = '/home/data1/group1/datasets/imagenet_val_FH_500_500/'
    sigma = 0.8
    kernel = 11
    K, min_size = 500, 500
    count = 0
    for c in os.listdir(dataset_root):
        c_root = os.path.join(dataset_root,c)
        os.makedirs(os.path.join(save_root, c),exist_ok=True)
        for i in os.listdir(c_root):
            count += 1
            # if i.replace('JPEG','npy') in  os.listdir(os.path.join(save_root, c)):
            #     continue
            img = cv2.imread(os.path.join(c_root, i))
            # skimage自带的felzenszwalb算法
            seg = felzenszwalb(img, scale=K, sigma=sigma, min_size=min_size)
            save_data = {'seg':seg.astype(np.uint8), 'num_parts': len(np.unique(seg))}
            np.save(os.path.join(save_root, c, i.replace('JPEG','npy')), save_data)
            print(count)
    exit()

    img  = '/home/user01/datasets/VOCdevkit/VOC2012/JPEGImages/2012_004331.jpg'
    # get_rect(img)
    image = cv2.imread(img)
    image = cv2.resize(image,(128,128))
    print("[INFO]: Calculating candidate region of interest using Selective Search ...")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()

    print("[INFO]: Found {} candidate region of interests".format(len(rects)))
    output = image.copy()
    for i in range(0, len(rects)):
        for (x, y, w, h) in rects[i:i]:
            color = [random.randint(0, 255) for j in range(0, 3)]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Deer Image", output)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        # close image show window
        cv2.destroyAllWindows()

    exit()
    img = cv2.imread(img)
    boxes = selective_search(img, img.shape[0], img.shape[1],res_size=(1280,760))
    print(boxes)