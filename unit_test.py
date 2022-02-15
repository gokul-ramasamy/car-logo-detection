from datagen import flatten, get_image_paths
from datagen import CarLogoDataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

#Testing the flatten function
output1 = flatten([[1,2,3], [4,5,6], [7,8],[9]])
print('>> Unit Test 1: flatten()')
print(output1)

#Testing the get_image_paths function
TRAIN_DATA_PATH = './dataset/Train' 
TEST_DATA_PATH = './dataset/Test'

output2 = get_image_paths(TRAIN_DATA_PATH, TEST_DATA_PATH)
print('\n>> Unit Test 2: get_image_paths()')
print("Train size: {} \nTest size: {}".format(len(output2[0]), len(output2[1])))
print("Classes: {}".format(output2[2]))

#Testing the dataset class with training data
train_image_paths, test_image_paths, class_to_idx = get_image_paths(TRAIN_DATA_PATH, TEST_DATA_PATH)
train_dataset = CarLogoDataset(train_image_paths, class_to_idx, transform = transforms.Compose([
                                                                                                transforms.Resize((256,256)),
                                                                                                transforms.ToTensor()
                                                                                                ]))


print('\n>> Unit Test 3: CarLogoDataset')
print('The shape of tensor for 50th image in train dataset: ',train_dataset[50][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[50][1])


plt_tensor = train_dataset[50][0].transpose(0,2).transpose(0,1)
plt.imshow(plt_tensor)
plt.title("50th image in the train dataset")
plt.show()

#Testing a DataLoader with the defined dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print("\n>> Unit Test 4: DataLoader")
print('The shape of the first batch: {}'.format(next(iter(train_loader))[0].shape))
