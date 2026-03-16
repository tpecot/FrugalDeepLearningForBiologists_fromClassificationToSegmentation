# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################


import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import time
from tempfile import TemporaryDirectory
import time
import os
import marimo as mo
from pathlib import Path
import datetime
from PIL import Image
import pandas as pd
import random


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms_with_augmentations = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5)),  # Random blur
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        transforms.RandomAutocontrast(p=0.1),
        transforms.RandomEqualize(p=0.1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# Normalization for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# Data normalization for inference
data_transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# ====================== Training ======================
def train_model(output_path, output_model_path, model, dataloaders, criterion, optimizer, scheduler, device, dataset_sizes, num_epochs=25):
    since = time.time()
    best_acc = 0.0

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
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

            if phase == 'train':
                scheduler.step(epoch_loss)

            # Store metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc.cpu())
            else:
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc.cpu())
        
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({'model_state_dict': model.state_dict()}, output_model_path)


    # Generate training plots
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path + '/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')


def train(output_path, data_dir, frozen_network=True, batch_size = 4, lr = 0.001, nb_epochs_without_improvement = 5, num_epochs = 100, augmentations=True):

    if augmentations:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms_with_augmentations[x])
                          for x in ['train', 'val']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, pin_memory=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    model_ft = models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
    # Freeze all layers except the final layer
    if frozen_network:
        for param in model_ft.parameters():
            param.requires_grad = False  # Freeze all layers

    # Modify the classifier head for your number of classes
    model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))


    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()


    # Use Adam optimizer with weight decay
    optimizer = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=nb_epochs_without_improvement, factor=0.5)

    # Decay LR by a factor of 0.1 every 7 epochs
    if augmentations:
        output_model_dir = output_path + "/ep_" + str(num_epochs) + "_bs_" + str(batch_size) + "_lr_" + str(lr) + "_with_aug_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        output_model_dir = output_path + "/ep_" + str(num_epochs) + "_bs_" + str(batch_size) + "_lr_" + str(lr) + "_no_aug_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    Path(output_model_dir).mkdir(parents=True, exist_ok=True)
    
    model_ft = train_model(output_model_dir, output_model_dir + '/best_model.pth', model_ft, dataloaders, criterion, optimizer, scheduler, device=device, dataset_sizes=dataset_sizes, num_epochs=num_epochs)

    
# ====================== Inference ======================
def preprocess_image(image_path, input_size=(224, 224)):
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Apply transforms
    img_tensor = data_transforms_test(img).unsqueeze(0)  # Add batch dimension

    return img_tensor

def predict(model_path, input_tensor, nb_classes, device):
    """
    Run inference on a single image or batch of images

    Args:
        model: PyTorch model
        input_tensor: Input tensor (1, C, H, W) or (N, C, H, W)
        device: Device to run inference on

    Returns:
        Predicted class probabilities and class indices
    """
    model = models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, nb_classes)


    # 2. Load the saved state dictionary
    checkpoint = torch.load(model_path, map_location=device)

    # 3. Load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval().to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_classes = torch.max(outputs, 1)

    return probabilities.cpu().numpy(), predicted_classes.cpu().numpy()

def inference_pipeline(model_path, image_dir, train_dir):
    """
    Complete inference pipeline for single image or batch of images

    Args:
        model: PyTorch model
        image_dir: directory path to image(s)
        train_dir: directory to training path

    Returns:
        Dictionary with predictions, probabilities, and processing time
    """
    start_time = time.time()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    image_list = os.listdir(image_dir)
    images = []
    for image in image_list:
        if image.endswith(('.jpg', '.png')):
            images.append(image)
    
    classes = os.listdir(train_dir)
    results = np.zeros((len(images), len(classes)))
    current_image = 0
    for image in images:
        input_tensor = preprocess_image(image_dir + '/' + image)
        results[current_image, :], predicted_class = predict(model_path, input_tensor, len(classes), device)
        current_image += 1
    
    # convert array into dataframe
    DF = pd.DataFrame(results, images, classes)
    # save the dataframe as a csv file
    DF.to_csv(image_dir + '/' + "results.csv")