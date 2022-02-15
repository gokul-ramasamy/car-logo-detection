import numpy as np
import glob
from torch.utils.data import Dataset
from skimage import io
from PIL import Image


#Function to flatten the list of lists
def flatten(input_list):
    output_list = [item for sublist in input_list for item in sublist]
    return output_list

#Function to get the train image paths and test image paths in a list
def get_image_paths(train_data_path, test_data_path, shuffle=False): 
    train_image_paths = list() #Storing image paths in a list
    classes = list() #store class variables

    #Storing the train image paths and classes in their respective list
    for data_path in glob.glob(train_data_path + '/*'):
        classes.append(data_path.split('/')[-1]) 
        train_image_paths.append(glob.glob(data_path + '/*'))

    #Flatenning the list of lists
    train_image_paths = flatten(train_image_paths)
    #Shuffle if only true is passed
    if shuffle == True:
        np.random.shuffle(train_image_paths)

    # print('train_image_path example:', train_image_paths[0])
    # print('class example:', train_image_paths[0].split('/')[-2])

    #Splitting the data into training and validation    
    train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 


    #Storing the test image paths in their respective lists
    test_image_paths = list()
    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = flatten(test_image_paths)

    # print("Train size: {} \nTest size: {}".format(len(train_image_paths), len(test_image_paths)))

    #Converting the classes to idx
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    return train_image_paths, valid_image_paths, test_image_paths, class_to_idx

#Defining the Car Logo Dataset Class
class CarLogoDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, transform=False):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert('RGB')

        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
        
        return image, label
