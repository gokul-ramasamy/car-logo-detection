# Car Logo Detection
The repository contains a simple code to train and test a pretrained model (here VGG16) for Car Logo Detection.

Here is the [link](https://drive.google.com/drive/folders/1j7rA5rIBhdTGW-oyfs1_zsfOp5xO0xhd) to the Car Logo Detection Dataset, the dataset contains 18 well-known car brands as its labels. The labels are the name of the subdirectories. The train set consists of 50 images per class totaling to 900 images and the test set contains 100 images.

The dataset should be downloaded and placed in the same environment. Also, note that the codebase is in PyTorch. So you need your PyTorch set up in your environment, the model can train both on CPUs and GPUs.

To run the training please use the following command
```
python main.py
```

The code also logs the training loss and validation in tensorboard

The following will be added later
1. Ability to train the model from a saved checkpoint
2. Separate train and inference code
3. JSON for parameters
