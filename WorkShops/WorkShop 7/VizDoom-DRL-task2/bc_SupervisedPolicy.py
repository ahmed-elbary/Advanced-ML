############################################################################
# bc_SupervisedPolicy.py
#
# This program trains a supervised policy to be used as a behaviour cloning 
# agent to play the game of VizDoom. Prior to training, it verifies
# whether data pre-processing is required. If that's the case, it takes 
# single image data in order to generate data containing 4-image sequences.
# The method preprocess_image_data() does the image pre-processing and the 
# method run_experiment() trains the classifier and saves the model. 
#
# This program works together with sb_VizDoom_FromDemonstration.py -- look
# at the workshop description of this week for further information.
# 
# Notes:
# The source folder should exist with subfolders -- from collected data.
# The target folder should not exist for it to create 4-image instances, 
# as a pre-processing step before training the classifier.
#
# Contact: hcuayahuitl@lincoln.ac.uk
# Last update on 12 March 2025.
############################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import pathlib
import os
import cv2
import time

# Convert individual images into sequences of images
def preprocess_image_data(datapath_source, datapath_target):
    image_data = {}  # full path file -> img
    file_paths = {}  # file name -> full path file
    labels = []

    # Load images from the source folder (datapath_source)
    print("READING image data from " + str(datapath_source))
    for folder in os.listdir(datapath_source):
        labels.append(folder)
        check_create_folder(datapath_target + '/' + folder)
        for file in os.listdir(datapath_source + '/' + folder):
            full_file_path = datapath_source + '/' + folder + '/' + file
            img = cv2.imread(full_file_path)
            img_id = file[2:]
            img_id = float(img_id.replace(".jpg", ""))
            image_data[full_file_path] = img
            file_paths[img_id] = full_file_path

    # Sort images by IDs and create sequences of images
    print("GENERATING stacked images in " + str(datapath_target))
    list_img_ids = sorted(file_paths.keys())
    for i in range(0, len(list_img_ids)):
        if i + 4 + 1 >= len(list_img_ids):
            break
        sublistOf4 = list_img_ids[i:i + 4]
        target_img_id = list_img_ids[i + 4 + 1]
        target_img_label = get_image_label_from_id(target_img_id, file_paths, labels)

        # Save images in the target folder (datapath_target)
        imgs = [image_data[file_paths[img_id]] for img_id in sublistOf4]
        joined_imgs = np.concatenate(imgs, axis=1)
        target_file_path = file_paths[sublistOf4[0]].replace(datapath_source, datapath_target)
        cv2.imwrite(target_file_path, joined_imgs)

def verify_datapaths(datapath_source, datapath_target):
    if not os.path.exists(datapath_source):
        print(f"datapath_source={datapath_source} does not exist! Please revise.")
        exit(0)

    if os.path.exists(datapath_target):
        print(f"datapath_target={datapath_target} exists already! NOTHING to process...")
        return False
    else:
        check_create_folder(datapath_target)
        return True

def check_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"path={path} has been created!")

def get_image_label_from_id(img_id, file_paths, labels):
    file_path = file_paths[img_id]
    for label in labels:
        file_to_look_for = label + '_' + str(img_id) + '.jpg'
        if file_path.endswith('/' + file_to_look_for):
            return label
    print(f"ERROR: couldn't find label for img_id={img_id} in data structure file_paths")
    exit(0)

# Load dataset from the target folder
def read_image_data(data_path, verbose, input_shape):
    print("READING DATA...")

    transform = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    X, Y = [], []
    for images, labels in dataloader:
        X.append(images)
        Y.append(labels)

    X = torch.cat(X)
    Y = torch.cat(Y)
    
    print(f"X={X.shape}")
    print(f"Y={Y.shape}")
    print(f"class_names={dataset.classes}")

    return X, Y, dataset.classes

# Define the CNN classifier model
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.elu1 = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.elu2 = nn.ELU()
        self.flatten = nn.Flatten()

        # Compute the output size after convolutions (dynamically)
        conv_out_size = self._get_conv_output_size(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 64)
        self.norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, shape[0], shape[1])  # (batch_size, channels, height, width)
            x = self.pool(self.elu1(self.conv1(sample_input)))
            x = self.pool(self.elu2(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(self.elu1(self.conv1(x)))
        x = self.pool(self.elu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.norm(self.fc1(x))
        #x = self.softmax(self.fc2(x)) # remove this
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    print(f'TRAINING model')
    model.train()
    
    # Track the overall loss for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)  # move inputs to the GPU
            labels = labels.to(device)  # move labels to the GPU
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print progress every X batches
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')
        
        # Print average loss for the epoch
        avg_loss = running_loss / total_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}')

def evaluate_model(model, test_loader, device):
    print(f'EVALUATING model')
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # move inputs to the same device
            labels = labels.to(device)  # move labels to the same device
            
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()

    total_samples = sum(len(batch[1]) for batch in test_loader) 
    accuracy = total_correct / total_samples
    print(f'Accuracy: {accuracy:.4f}')

def run_experiment(data_path, model_name, BATCH_SIZE, EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    INPUT_SHAPE = (84, 84 * 4, 3)
    x_train, y_train, class_names = read_image_data(data_path, False, INPUT_SHAPE)
    if model_name == None: return None, class_names
    NUM_CLASSES = len(class_names)
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 20

    print("CREATING classifier model...")
    model = ImageClassifier(NUM_CLASSES, INPUT_SHAPE).to(device)
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    train_model(model, dataloader, criterion, optimiser, device, num_epochs=NUM_EPOCHS)
    evaluate_model(model, dataloader, device)

    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")
    
    return model, class_names

def load_pretrained_classifier(datapath_train, model_name):
    print("LOADING pre-trained classifier", model_name)
    _, class_names = run_experiment(datapath_train, None, None, None)
    model = ImageClassifier(len(class_names), (84, 84 * 4, 3))
    model.load_state_dict(torch.load(model_name))
    model.eval()
    print(f"Supervised Pre-Trained model={model_name} loaded...")
    return model, class_names

datapath_source = "./vizdoom-1img"
datapath_target = "./vizdoom-4img"

model_name = "VizDoom-SPT.pth"
EPOCHS = 20
BATCH_SIZE = 32

if __name__ == "__main__":
    if verify_datapaths(datapath_source, datapath_target):
        preprocess_image_data(datapath_source, datapath_target)
    run_experiment(datapath_target, model_name, BATCH_SIZE, EPOCHS)
