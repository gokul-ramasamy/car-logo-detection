from datagen import get_image_paths
from datagen import CarLogoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import torch
import time
from torchsummary import summary
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

#Global Variables (can be moved to a JSON file)
#Testing the get_image_paths function
TRAIN_DATA_PATH = './dataset/Train' 
TEST_DATA_PATH = './dataset/Test'
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
INPUT_IMG_SIZE = (224,224)
MODEL_SAVE_PATH = './model/'
LOG_ROOT_DIR = './logs/'

#Start time
start_time = time.time()

#Create model save directory if not available
if not os.path.isdir(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

#Create model save directory if not available
if not os.path.isdir(LOG_ROOT_DIR):
    os.mkdir(LOG_ROOT_DIR)

#Defining the training dataset
train_image_paths, valid_image_paths, test_image_paths, class_to_idx = get_image_paths(TRAIN_DATA_PATH, TEST_DATA_PATH)
train_dataset = CarLogoDataset(train_image_paths, class_to_idx, transform = transforms.Compose([
                                                                                                transforms.Resize(INPUT_IMG_SIZE),
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Normalize((0.5650,0.5381,0.5478), (0.2725,0.2781,0.2705))
                                                                                                
                                                                                                ]))

valid_dataset = CarLogoDataset(valid_image_paths, class_to_idx, transform = transforms.Compose([
                                                                                                transforms.Resize(INPUT_IMG_SIZE),
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Normalize((0.5650,0.5381,0.5478), (0.2725,0.2781,0.2705))
                                                                                                
                                                                                                ]))
test_dataset = CarLogoDataset(test_image_paths, class_to_idx, transform = transforms.Compose([
                                                                                                transforms.Resize(INPUT_IMG_SIZE),
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Normalize((0.4660,0.4432,0.4654), (0.2458,0.2546,0.2471))
                                                                                                ]))

#Defining the data loader
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

#Defining the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading the pret-trained VGG-16 model and pushing the model to device
model = models.vgg16(pretrained=True)

#Redefining the last layer
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=len(list(class_to_idx.keys())))
# summary(model, (3,224,224))

#Pushing model to GPU
model = model.to(device)
#Loss function and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
							lr = 1e-4,
							weight_decay = 1e-8)

#Create a checkpoint path to save the checkpoints
CHECKPOINT_PATH = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'/' 
if not os.path.isdir(MODEL_SAVE_PATH+CHECKPOINT_PATH):
    os.mkdir(MODEL_SAVE_PATH+CHECKPOINT_PATH)

#Create a log folder to log the training parameters such as loss
#Log directory
LOG_DIR = LOG_ROOT_DIR + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'/'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

#Creates a file writer for the log directory (loss)
loss_writer = SummaryWriter(LOG_DIR+'loss')

#define training function
def train (model, loader, criterion, gpu):
    model.train()
    current_loss = 0
    current_correct = 0
    for train, y_train in iter(loader):
        if gpu:
            train, y_train = train.to(device), y_train.to(device)
        optimizer.zero_grad()
        output = model.forward(train)
        _, preds = torch.max(output,1)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()*train.size(0)
        current_correct += torch.sum(preds == y_train.data)
    epoch_loss = current_loss / len(train_loader.dataset)
    epoch_acc = current_correct.double() / len(train_loader.dataset)
        
    return epoch_loss, epoch_acc

#define validation function
def validation (model, loader, criterion, gpu):
    model.eval()
    valid_loss = 0
    valid_correct = 0
    for valid, y_valid in iter(loader):
        if gpu:
            valid, y_valid = valid.to(device), y_valid.to(device)
        output = model.forward(valid)
        valid_loss += criterion(output, y_valid).item()*valid.size(0)
        equal = (output.max(dim=1)[1] == y_valid.data)
        valid_correct += torch.sum(equal)#type(torch.FloatTensor)
    
    epoch_loss = valid_loss / len(test_loader.dataset)
    epoch_acc = valid_correct.double() / len(test_loader.dataset)
    
    return epoch_loss, epoch_acc


#Initialize training params  
#freeze gradient parameters in pretrained model
for param in model.parameters():
    param.require_grad = False
#train and validate
epochs = 25
epoch = 0

    
for e in range(epochs):
    epoch +=1
    print('>> Epoch {}'.format(epoch))
    #TRAINING
    with torch.set_grad_enabled(True):
        epoch_train_loss, epoch_train_acc = train(model,train_loader, loss_fn, torch.cuda.is_available())
        
        #Saving checkpoints after every epoch
        torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': epoch_train_loss,
		}, MODEL_SAVE_PATH+CHECKPOINT_PATH+'model.pth')
        #Logging the loss 
        loss_writer.add_scalar('Training Loss', epoch_train_loss, epoch)

    print("Epoch: {} Train Loss : {:.4f}  Train Accuracy: {:.4f}".format(epoch,epoch_train_loss,epoch_train_acc))
    
    #VALIDATION
    with torch.no_grad():
        epoch_val_loss, epoch_val_acc = validation(model, valid_loader, loss_fn, torch.cuda.is_available())
        loss_writer.add_scalar('Validation Loss', epoch_val_loss, epoch)
    print("Epoch: {} Validation Loss : {:.4f}  Validation Accuracy {:.4f}".format(epoch,epoch_val_loss,epoch_val_acc))


print("Total time taken = {}".format(time.time()-start_time))

###TESTING
model.eval()
total = 0
correct = 0 
count = 0
#iterating for each sample in the test dataset once
for test, y_test in iter(test_loader):
    test, y_test = test.to('cuda'), y_test.to('cuda')
#Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(test)
        ps = torch.exp(output)
        _, predicted = torch.max(output.data,1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()     
        count += 1
        print("Accuracy of network on test images is ... {:.4f}....count: {}".format(100*correct/total,  count ))
