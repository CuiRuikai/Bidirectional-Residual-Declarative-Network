#
# tools for dataset pre-processing
#

import os
import shutil
import cv2
import numpy as np


def mkdir(new_folder):
    f = os.path.exists(new_folder)
    if not f:
        os.makedirs(new_folder)
        return True
    else:
        print("---  folder exist " + new_folder + " ---")
        return False


def crop_image(face_detector, origin, target, img_name, meta,no, emotion, width=128):
    '''
    Detect face and crop that region
    :param face_detector: As name
    :param origin: source image name
    :param target: target image name
    :param width: width of region/2
    :return: None
    '''
    img = cv2.imread(origin)
    rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    thresh = 0.5
    boxes = face_detector.predict(rgb_img, thresh)
    index=0
    for i in range(len(boxes)):
        (x, y, w, h, p) = boxes[i]
        if (int(y - width)) > 0 and (int(y + width)) < img.shape[0] and (int(x - width)) > 0 and (int(x + width)) < \
                img.shape[1]:
            mask = img[(int(y - width)):(int(y + width)), (int(x - width)):(int(x + width)), :]
            cv2.imwrite(target[:-4] + "_" + str(index) + ".png", mask)
            meta['no'].append(no)
            meta['filename'].append(img_name[:-4] + "_" + str(index) + ".png")
            meta['prob'].append(p)
            meta['emotion'].append(emotion)
            index += 1
    return meta

def train_test_split(source_path, target_path, ratio):
    '''
    Split the dataset
    :param ratio: split ratio
    :return: None
    '''
    mkdir(target_path)
    mkdir(target_path + 'train/')
    mkdir(target_path + 'val/')
    folders = os.listdir(source_path)
    for folder in folders:
        target_subfolder_train = target_path +'train/' + folder + '/'
        target_subfolder_test = target_path + 'val/' + folder + '/'
        source_subfolder = source_path + folder + '/'
        mkdir(target_subfolder_train)
        mkdir(target_subfolder_test)
        files = os.listdir(source_subfolder)
        samples = np.random.choice(len(files), int(len(files) * ratio))
        for i in range(len(files)):
            if i in samples:
                shutil.copyfile(source_subfolder + files[i], target_subfolder_test + files[i])
            else:
                shutil.copyfile(source_subfolder + files[i], target_subfolder_train + files[i])
    return None

def image_filp(path):
    '''
    Flip every images
    :return: None
    '''
    folders = os.listdir(path)
    for folder in folders:
        files = os.listdir(path+'/'+ folder)
        for name in files:
            origin = path + '/' + folder + '/' + name
            target = path + '/' + folder + '/' + name[:-4] + "_flip" + ".png"
            img = cv2.imread(origin)
            img_flip = cv2.flip(img, 1)
            cv2.imwrite(target, img_flip)
        print("finish " + folder)
    return None

def image_ressize(path):
    '''
    Resize images
    :return: None
    '''
    folders = os.listdir(path)
    for folder in folders:
        files = os.listdir(path+'/'+ folder)
        for name in files:
            origin = path + '/' + folder + '/' + name
            target_width = path + '/' + folder + '/' + name[:-4] + "_width" + ".png"
            target_hight = path + '/' + folder + '/' + name[:-4] + "_hight" + ".png"
            img = cv2.imread(origin)
            img_width = img_test2 = cv2.resize(img, (0, 0), fx=1.5, fy=1, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target_width, img_width)
            img_hight = img_test2 = cv2.resize(img, (0, 0), fx=1, fy=1.5, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target_hight, img_hight)
        print("finish " + folder)
    return None