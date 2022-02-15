from datagen import get_image_paths
from PIL import Image
import numpy as np
from tqdm import tqdm

#Global Variables
#Testing the get_image_paths function
TRAIN_DATA_PATH = './dataset/Train' 
TEST_DATA_PATH = './dataset/Test'
INPUT_IMG_SIZE = (256,256)

#Variables that are going to be used across functions
train_image_paths, valid_image_paths, test_image_paths, class_to_idx = get_image_paths(TRAIN_DATA_PATH, TEST_DATA_PATH)

#Getting the image paths
def channel_diff(image_paths, write_directory):
    
    other_channel_imgs = list()
    alpha_imgs = list()
    other_imgs = list()
    grayscale_imgs = list()

    for a_path in tqdm(image_paths):
        im = Image.open(a_path)
        im = im.resize(INPUT_IMG_SIZE)
        im_array = np.array(im)
        n = im_array.shape[-1]

        if len(im_array.shape) == 3:

            if n==4:
                print(a_path+ '-alpha channel')
                alpha_imgs.append(a_path+'\n')
            
            if n != 3 and n != 4:
                print(a_path + '- Others')
                other_imgs.append(a_path+'\n')
        else:
            print(a_path+ '- Grayscale')
            grayscale_imgs.append(a_path+'\n')

    #Writing all these to a text file
    with open(write_directory+'grayscale_imgs.txt', 'w+') as f:
        f.writelines(grayscale_imgs)
    with open(write_directory+'alpha_imgs'+'.txt', 'w+') as f:
        f.writelines(alpha_imgs)
    with open(write_directory+'other_imgs'+'.txt', 'w+') as f:
        f.writelines(other_imgs)
    with open(write_directory+'other_channel_imgs'+'.txt', 'w+') as f:
        f.writelines(other_channel_imgs)
    
    return None

#Calculating the average, maximum and minimum image size
def avg_image_size(image_paths, txt_write_path):
    mean_h = 0
    mean_w = 0
    all_h = list()
    all_w = list()
    area = list()


    for a_path in tqdm(image_paths):
        im = Image.open(a_path)
        h, w = im.size
        all_h.append(h)
        all_w.append(w)
        area.append(h*w)
        mean_h += h
        mean_w += w
    
    mean_h /= len(image_paths)
    mean_w /= len(image_paths)

    max_h, max_w = all_h[area.index(max(area))], all_w[area.index(max(area))]
    min_h, min_w = all_h[area.index(min(area))], all_w[area.index(min(area))]

    with open(txt_write_path, 'w+') as f:
        f.writelines('Average height = {}\n'.format(mean_h))
        f.writelines('Average width = {}\n'.format(mean_w))
        f.writelines('Minimum height = {}\n'.format(min(all_h)))
        f.writelines('Maximum height = {}\n'.format(max(all_h)))
        f.writelines('Minimum width = {}\n'.format(min(all_w)))
        f.writelines('Maximum width = {}\n'.format(max(all_w)))
        f.writelines('Largest image = {}\n'.format((max_h, max_w)))
        f.writelines('Smallest image = {}\n'.format((min_h, min_w)))

    
    return None

#File formats present
def formats(image_paths):
    formats = list()
    for a_path in image_paths:
        if a_path.split('.')[-1] not in formats:
            formats.append(a_path.split('.')[-1])
    print(formats)

    return None

#Counting the number of jpgs and pngs from the dataset
def jpg_png_count(image_path, write_path):
    jpg_count = 0
    png_count = 0
    other_count = 0
    for a_path in tqdm(image_path):
        if a_path.split('.')[-1] == 'jpg' or a_path.split('.')[-1] == 'jpeg':
            jpg_count += 1
        elif a_path.split('.')[-1] == 'png':
            png_count += 1
        else:
            other_count += 1
    
    with open(write_path, 'w+') as f:
        f.writelines('Number of jpeg images = {}\n'.format(jpg_count))
        f.writelines('Number of png images = {}\n'.format(png_count))

    return None

avg_image_size(test_image_paths, "./eda/test_hw")
