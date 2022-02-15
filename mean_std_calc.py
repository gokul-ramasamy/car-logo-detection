from datagen import get_image_paths
from datagen import CarLogoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

#Global Variables
#Testing the get_image_paths function
TRAIN_DATA_PATH = './dataset/Train' 
TEST_DATA_PATH = './dataset/Test'
INPUT_IMG_SIZE = (256,256)

#Defining the training dataset
train_image_paths, test_image_paths, class_to_idx = get_image_paths(TRAIN_DATA_PATH, TEST_DATA_PATH)
train_dataset = CarLogoDataset(train_image_paths, class_to_idx, transform = transforms.Compose([
                                                                                                transforms.Resize(INPUT_IMG_SIZE),
                                                                                                transforms.ToTensor()
                                                                                                ]))
test_dataset = CarLogoDataset(test_image_paths, class_to_idx, transform = transforms.Compose([
                                                                                                transforms.Resize(INPUT_IMG_SIZE),
                                                                                                transforms.ToTensor()
                                                                                                ]))                                                                                                

#Defining the data loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#Initialising mean, std and total number of images
mean = 0
std = 0
nb_samples = 0

for batch in tqdm(test_loader):
    data = batch[0]
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1) #Flattening the tensor
    mean += data.mean(2).sum(0) #Calculate mean along the last dimension and then sum the mean across all images in the batch
    std += data.std(2).sum(0) #Calculate std along the last dimension and then sum the std across all images in the batch
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print("The mean is {}".format(mean))
print("The std is {}".format(std))