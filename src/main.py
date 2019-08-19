"""
2019/08/19
devkim1102@gmail.com
"""

import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms, datasets

import src.config as con
import src.network_vgg as network_vgg


class cat_dog(Dataset):

    def __init__(self, root, file_list, dataset_size, transform = None):
        self.root = root
        self.file_list = file_list
        self.dataset_size = dataset_size
        self.transform = transform
        samples = []

        for file in self.file_list:
            image_path = os.path.join(self.root, file)
            samples.append(image_path)

        self.samples = samples

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        path = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f)
            original = img.convert('RGB')

        if self.transform:
            transformed = self.transform(original)

        return path, transformed

def loadModel(model, path_pretrainedModel, device):
    pretrained_dict = torch.load(path_pretrainedModel, map_location=device)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def train_model(model, criterion, optimizer, scheduler, device, num_epochs = 10):

    # data pre processing
    # image data should be transform into tensor format before feed in network

    train_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(con.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(con.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])}

    train_dataset = datasets.ImageFolder(root=con.train_data_dir, transform=train_transforms['train'])
    train_loader = torch.utils.data.DataLoader \
        (train_dataset, batch_size=con.batch_size, shuffle=True, num_workers=8)
    val_dataset = datasets.ImageFolder(root=con.val_data_dir, transform=train_transforms['val'])
    val_loader = torch.utils.data.DataLoader\
        (val_dataset, batch_size=con.batch_size, shuffle=True, num_workers=8)

    dataloaders = {'train':train_loader, 'val':val_loader}

    # length of train dataset
    dataset_sizes = {'train':len(train_dataset), 'val':len(val_dataset)}

    # name list of classes
    class_names = train_dataset.classes
    print(class_names)

    # load the model
    try:
        model_file = os.path.join(con.model_path, os.listdir(con.model_path)[0])
        model = loadModel(model, model_file, device)
        print("pre-trained model found... load the model")
    except:
        print("pre-trained model not found...")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # training time checkpoint:start
    since = time.time()

    for epoch in range(num_epochs):
        print(con.CRED+'\n===> epoch %d' % epoch + con.CEND)
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.3f} Acc: {:.3f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since

    print(con.CRED + "===>Training END" + con.CEND)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # store best model's state_dict
    if num_epochs > 0:
        fname = con.model_path + "epoch_" + str(epoch)
        torch.save(best_model_wts, fname)
    return model

def test_model(model, device, class_names):

    # data pre processing
    # image data should be transform into tensor format before feed in network

    test_transforms = transforms.Compose([
        transforms.RandomResizedCrop(con.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    file_list = os.listdir(con.test_data_dir)
    dataset_size = len(file_list)

    test_dataset = cat_dog(root=con.test_data_dir, file_list=file_list, dataset_size=dataset_size, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=con.batch_size, shuffle=True, num_workers=8)

    # make a sub-dirs for each classes
    for class_name in class_names:
        dir_name = os.path.join(con.log_path, class_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    log_count = 0

    for original_path, inputs in test_loader:
        inputs = inputs.to(device)

        with torch.no_grad(): # model parameter will not be updating
            outputs = model(inputs)
            _, predicts = torch.max(outputs, 1) # prediction

            for i in range(list(predicts.size())[0]):
                label = class_names[predicts[i].item()]

                with open(original_path[i], 'rb') as f:
                    img = Image.open(f)
                    original_img = img.convert('RGB')

                # store classified image
                img_name = con.log_path + "/" + label + "/" + str(log_count) + ".png"
                original_img.save(img_name)
                log_count += 1
                if log_count % 100 == 0:
                    print("==> Number of proccessed file: ", log_count)
    print(con.CRED+ "===> Test over" + con.CEND)

def run():
    print("running...")

    # step1
    # construct network, move to gpu device

    network_model = network_vgg.VGG(num_classes=con.class_num)

    # use gpu if there are gpu device detected
    device = torch.device(con.device_param if torch.cuda.is_available() else "cpu")

    network_model = network_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network_model.parameters(), lr=con.learning_rate)

    # Decay learning rate by a factor of 0.1 every 25 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # construct model directory to record model state_dict
    if not os.path.exists(con.model_path):
        os.makedirs(con.model_path, mode=0o777)

    network_model = train_model(network_model, criterion, optimizer, scheduler, device, num_epochs=con.num_epochs)

    # step2
    # test the network with test data

    # construct log directory to record failed images
    class_names = os.listdir(con.train_data_dir)

    if not os.path.exists(con.log_path):
        os.makedirs(con.log_path, mode=0o777)
    else:
        for dir in os.listdir(con.log_path):
            dir_path = os.path.join(con.log_path, dir)
            file_list = os.listdir(dir_path)
            for file in file_list:
                file_path = os.path.join(dir_path, file)
                os.unlink(file_path)
            os.rmdir(dir_path)

    test_model(network_model, device, class_names)

if __name__ == '__main__':
    run()