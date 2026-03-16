# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tifffile
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.transform import resize
import albumentations as A
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import MulticlassJaccardIndex
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import json

def create_json(classifier_name, converted_model, dim, nb_channels, channel_def, pixel_size, class_labels, model_path):
    data = {"pixel_classifier_type": "OpenCVPixelClassifier", "metadata": {"inputPadding": 0, "inputResolution": {
        "pixelWidth": {"value": pixel_size, "unit": "µm"}, "pixelHeight": {"value": pixel_size, "unit": "µm"},
        "zSpacing": {"value": 1.0, "unit": "z-slice"}, "timeUnit": "SECONDS", "timepoints": []}, "inputWidth": dim,
                                                                           "inputHeight": dim, "inputNumChannels": nb_channels,
                                                                           "outputType": "PROBABILITY",
                                                                           "outputPixelType": "FLOAT32",
                                                                           "outputChannels": [],
                                                                           "classificationLabels": class_labels},
            "op": {"type": "data.op.channels", "colorTransforms": channel_def,
                   "op": {"type": "op.core.sequential", "ops": [{"type": "op.core.convert", "pixelType": "FLOAT32"},
                                                                {"type": "op.ml.opencv-dnn",
                                                                 "model": {"dnn_model": "DjlDnnModel",
                                                                           "uris": [fr"file:///{converted_model}"],
                                                                           "engine": "PyTorch", "ndLayout": "NCHW",
                                                                           "inputs": {
                                                                               "input0": {"shape": [1, nb_channels, dim, dim]}},
                                                                           "outputs": {"output0": {
                                                                               "shape": [1, len(class_labels), dim, dim]}},
                                                                           "lazyInitialize": False}, "inputWidth": dim,
                                                                 "inputHeight": dim, "outputNames": [],
                                                                 "padding": {"x1": 0, "x2": 0, "y1": 0, "y2": 0}}]}}}

    with open(f"{model_path}/{classifier_name}.json", 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
        
def quantile_normalization(img):
    
    if len(img.shape) < 3:
        output = np.zeros((img.shape[0], img.shape[1]), 'float32')
        
        high = np.percentile(img, 99.8)
        low = np.percentile(img, 1)

        output = np.minimum(high, img)
        output = np.maximum(low, output)
        output = (output - low) / (high - low + 0.001)

    else:
        output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), 'float32')
        for i in range(img.shape[2]):
            high = np.percentile(img, 99.8)
            low = np.percentile(img, 1)
            
            output[:,:,i] = np.minimum(high, img[:,:,i])
            output[:,:,i] = np.maximum(low, output[:,:,i])
            output[:,:,i] = (output[:,:,i] - low) / (high - low)
    
    return output
    
def min_max_normalization(img):

    if len(img.shape) < 3:
        output = np.zeros((img.shape[0], img.shape[1]), 'float32')
        
        high = np.max(img)
        low = np.min(img)

        output = (img - low) / (high - low + 0.001)

    else:
        output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), 'float32')
        for i in range(img.shape[2]):
            high = np.max(img[:,:,i])
            low = np.min(img[:,:,i])
            
            output[:,:,i] = (img[:,:,i] - low) / (high - low + 0.001)
    
    return output

# ====================== Global Collate Function ======================
def collate_fn(batch):
    """Global collate function to avoid pickling issues"""
    images = torch.stack([item[0] for item in batch], dim=0)  # (batch, n_channels, H, W)
    masks = torch.stack([item[1] for item in batch], dim=0)    # (batch, H, W)
    return images, masks
    
# ====================== U-Net Model (parameterizable depth) ======================
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, depth=3, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth

        if depth < 2 or depth > 8:
            raise ValueError("Depth must be between 2 and 5")

        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        # Create encoder and decoder layers
        num_filters = 64
        for i in range(depth - 1):  # depth-1 down/up layers
            self.down_layers.append(Down(num_filters, num_filters * 2))
            num_filters *= 2

        # Create up layers with proper channel dimensions
        num_filters = num_filters  # Current number of filters at the bottom
        for i in range(depth - 2, -1, -1):  # depth-2 up layers (since we don't need up after last)
            # The input channels for Up should be (num_filters + corresponding skip connection channels)
            # The skip connection channels are 64 * 2^i
            skip_channels = 64 * (2 ** i)
            self.up_layers.append(Up(num_filters + skip_channels, num_filters // 2, bilinear))
            num_filters = num_filters // 2

        # Final convolution
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x = self.inc(x)
        skip_connections = []

        # Downsampling
        for down in self.down_layers:
            skip_connections.append(x)
            x = down(x)

        # Upsampling with skip connections
        for i, up in enumerate(self.up_layers):
            x = up(x, skip_connections[-(i+1)])

        # Final output
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # The DoubleConv should expect in_channels channels (sum of skip + up)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

import torch
import torch.nn as nn

class RescaleNormalizer(nn.Module):
    """Normalizes input to [0, 1] range"""
    def forward(self, x):
        """
        Args:
            x: Input tensor (HWC or CHW format)
        Returns:
            Normalized tensor in [0, 1] range
        """
        if x.dim() == 3:  # HWC format
            x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)

        # Find min and max for each channel
        n, c, h, w = x.shape
        x_flat = x.view(n, c, -1)

        normalized = torch.zeros_like(x)
        for i in range(c):
            channel = x[:, i:i+1, :, :]
            channel_min = channel.min()
            channel_max = channel.max()
            normalized[:, i:i+1, :, :] = (channel - channel_min) / (channel_max - channel_min + 0.001)

        return normalized

class PreprocessingWrapper(nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model
        self.normalizer = RescaleNormalizer()

    def forward(self, x):
        x = self.normalizer(x)
        return self.unet(x)

def create_traced_model(model, nb_channels, image_size):
    
    #model.eval()
    wrapper = PreprocessingWrapper(model)
    wrapper.eval()

    # Create sample input for tracing
    sample_input = torch.randn(1, nb_channels, image_size, image_size)

    # Trace the wrapper
    #traced_model = torch.jit.trace(model, sample_input)
    traced_model = torch.jit.trace(wrapper, sample_input)

    return traced_model

def load_model(model_path, n_channels=2, n_classes=3, unet_depth=4, device='cuda'):
    """Load a saved U-Net model

    Args:
        model_path: Path to the saved model file (.pth)
        n_channels: Number of input channels (must match training)
        n_classes: Number of output classes (must match training)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded U-Net model
    """
    # 1. Create a new instance of the model with the same architecture
    model = UNet(n_channels=n_channels, n_classes=n_classes, depth=unet_depth)

    # 2. Load the saved state dictionary
    checkpoint = torch.load(model_path, map_location=device)

    # 3. Load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Move model to the specified device
    model = model.to(device)

    # 5. Set to evaluation mode if you're doing inference
    model.eval()

    return model

def calculate_class_weights(dataset, device):
    all_masks = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        all_masks.append(mask.numpy())

    all_masks = np.concatenate([m.flatten() for m in all_masks])
    class_weights = compute_class_weight('balanced', classes=np.unique(all_masks), y=all_masks)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Numerically stable Dice loss implementation that guarantees positive values
        Args:
            inputs: Tensor of shape (N, C, H, W) - logits
            targets: Tensor of shape (N, H, W) - class indices
        Returns:
            Dice loss (always positive)
        """
        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(inputs, dim=1)  # (N, C, H, W)

        # Convert targets to one-hot encoding
        targets_onehot = torch.zeros_like(probabilities)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)  # (N, C, H, W)

        # Calculate intersection and union for all classes simultaneously
        intersection = (probabilities * targets_onehot).sum(dim=(2, 3))  # (N, C)
        cardinality = probabilities.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))  # (N, C)

        # Calculate Dice coefficient with smoothing
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)  # (N, C)

        # Clamp to avoid numerical issues
        dice = torch.clamp(dice, 0.0, 1.0)

        # Calculate loss as 1 - mean(Dice)
        loss = 1.0 - dice.mean()

        # Final safety check (should never be needed with proper clamping)
        if loss < 0:
            print(f"Warning: Negative loss detected. Clamping to 0.")
            print(f"Intersection: {intersection}")
            print(f"Cardinality: {cardinality}")
            print(f"Dice scores: {dice}")
            loss = torch.clamp(loss, min=0.0)

        return loss

def get_augmentations():
    """Define augmentation pipeline"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),      # Random horizontal flip (50% chance)
        A.VerticalFlip(p=0.5),        # Random vertical flip (50% chance)
        A.Rotate(limit=30, p=0.5),     # Random rotation between -30 and 30 degrees
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Random blur
        #A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # Random noise
        #A.RandomBrightnessContrast(p=0.3),  # Additional augmentation
    ])
        
# ====================== TIFF Dataset Class with Configurable Channels ======================
class TIFFDataset(Dataset):
    def __init__(self, input_dir, n_channels=3, image_size=256, augment=False):
        self.input_dir = input_dir
        self.n_channels = n_channels
        self.image_size = image_size
        self.augment = augment
        if augment:
            self.augmentation = get_augmentations()

        # Get list of image files
        self.image_files = [f for f in os.listdir(input_dir + "/images/")
                          if f.endswith(('.tif', '.tiff'))]

        # Verify all images and masks exist
        self.valid_indices = []
        for i, img_file in enumerate(self.image_files):
            base_name = os.path.splitext(img_file)[0]
            mask_candidates = [
                f"{base_name}.tif",
                f"{base_name}.tiff",
                f"{base_name}_mask.tif",
                f"{base_name}_mask.tiff"
            ]

            mask_found = False
            for candidate in mask_candidates:
                if os.path.exists(os.path.join(input_dir + "/masks/", candidate)):
                    mask_found = True
                    break

            if mask_found:
                self.valid_indices.append(i)

        print(f"Found {len(self.valid_indices)} valid image-mask pairs")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_file = self.image_files[real_idx]
        img_path = os.path.join(self.input_dir + "/images/", img_file)

        # Load image
        image = tifffile.imread(img_path)
        if len(image.shape)>2:
            image = np.transpose(image, (1, 2, 0)) 

        # load corresponding mask
        base_name = os.path.splitext(img_file)[0]
        mask_candidates = [
            f"{base_name}.tif",
            f"{base_name}.tiff",
            f"{base_name}_mask.tif",
            f"{base_name}_mask.tiff"
        ]

        mask_path = None
        for candidate in mask_candidates:
            full_path = os.path.join(self.input_dir + "/masks/", candidate)
            if os.path.exists(full_path):
                mask_path = full_path
                break

        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_file}")

        # Load mask
        mask = tifffile.imread(mask_path)

        # Apply augmentations if enabled
        if self.augment:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to float32 and normalize
        image = min_max_normalization(image)

        # Handle different channel configurations
        if len(image.shape) == 2:
            # Grayscale image - expand to n_channels
            image = np.stack([image] * self.n_channels, axis=-1)
        elif image.shape[-1] != self.n_channels:
            # If channels don't match, take first n_channels
            if image.shape[-1] < self.n_channels:
                # If fewer channels than needed, pad with zeros
                pad = np.zeros((image.shape[0], image.shape[1], self.n_channels - image.shape[-1]))
                image = np.concatenate([image, pad], axis=-1)
            else:
                # If more channels than needed, take first n_channels
                image = image[..., :self.n_channels]

        

        # Convert mask to long tensor
        mask = mask.astype(np.int64)

        # Resize if needed
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = self.resize_image(image)
            mask = self.resize_mask(mask)

        # Convert to tensors with correct shapes:
        # image: (H, W, n_channels) -> (n_channels, H, W) after permute
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (n_channels, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)

        return image, mask

    def resize_image(self, image):
        return resize(image, (self.image_size, self.image_size), preserve_range=True)

    def resize_mask(self, mask):
        resized = resize(mask, (self.image_size, self.image_size),
                        order=0, preserve_range=True, anti_aliasing=False)
        return np.rint(resized).astype(np.int64)

# ====================== Training Function (continued) ======================
def train_unet(model_path, classifier_name, class_labels, pixel_size, channel_def, training_dir, validation_dir, test_size=0.2, unet_depth=3, image_size=256, n_channels=3, n_classes=3, epochs=20, batch_size=4, learning_rate=1e-4, augmentations=True, early_stopping=20, nb_epochs_without_improvement=5):
    # Create dataset
    if validation_dir==None:
        try:
            dataset = TIFFDataset(training_dir, n_channels=n_channels, image_size=image_size, augment=augmentations)
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return None

        if len(dataset) == 0:
            print("No valid image-mask pairs found. Check your data paths.")
            return None
    
        # Split into train and validation
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

    else:
        try:
            training_dataset = TIFFDataset(training_dir, n_channels=n_channels, image_size=image_size, augment=augmentations)
        except Exception as e:
            print(f"Error creating training dataset: {e}")
            return None

        if len(training_dataset) == 0:
            print("No valid image-mask pairs found in training dataset. Check your data paths.")
            return None

        try:
            validation_dataset = TIFFDataset(validation_dir, n_channels=n_channels, image_size=image_size, augment=False)
        except Exception as e:
            print(f"Error creating validation dataset: {e}")
            return None

        if len(validation_dataset) == 0:
            print("No valid image-mask pairs found in validation dataset. Check your data paths.")
            return None
    
        train_dataset = training_dataset
        val_dataset = validation_dataset

    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Initialize model with correct number of input channels
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = UNet(n_channels=n_channels, n_classes=n_classes, depth=unet_depth).to(device)

    # Initialize with proper weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # Loss and optimizer
    class_weights = calculate_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=nb_epochs_without_improvement, factor=0.5)

    # Initialize metrics - use MulticlassJaccardIndex for Dice
    dice_metric = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    dice_loss = DiceLoss(num_classes=2, smooth=1e-6)

    
    # Training loop
    best_val_dice = 0.0

    # At the start of training
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    if augmentations:
        output_path = model_path + "/depth_" + str(unet_depth) + "_is_" + str(image_size) + "_ch_" + str(n_channels) + "_cl_" + str(n_classes) + "_ep_" + str(epochs) + "_bs_" + str(batch_size) + "_lr_" + str(learning_rate) + "_with_aug_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        output_path = model_path + "/depth_" + str(unet_depth) + "_is_" + str(image_size) + "_ch_" + str(n_channels) + "_cl_" + str(n_classes) + "_ep_" + str(epochs) + "_bs_" + str(batch_size) + "_lr_" + str(learning_rate) + "_no_aug_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []
    
    epochs_no_improve = 0
    early_stop = False
    for epoch in range(epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
            
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, masks in train_loop:
            try:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                # Calculate losses
                ce_loss = criterion(outputs, masks)
                dsc_loss = dice_loss(outputs, masks)
                loss = ce_loss + dsc_loss  
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_dice += dice_metric(outputs, masks).item()
                train_loop.set_postfix(loss=loss.item(), dice=dice_metric(outputs, masks).item())

            except Exception as e:
                print(f"Error during training: {e}")
                continue
                

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            for images, masks in val_loop:
                try:
                    images = images.to(device)
                    masks = masks.to(device)

                    outputs = model(images)
                    
                    # Calculate losses
                    ce_loss = criterion(outputs, masks)
                    dsc_loss = dice_loss(outputs, masks)
                    loss = ce_loss + dsc_loss
                    val_loss += loss.item()
                    val_dice += dice_metric(outputs, masks).item()
                    
                    val_loop.set_postfix(loss=loss.item(), dice=dice_metric(outputs, masks).item())
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue

        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_scores.append(train_dice)
        val_dice_scores.append(val_dice)
        
        # Update learning rate
        scheduler.step(val_loss)
        

        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        # Early stopping check
        # Save best model based on validation Dice score
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'dice': val_dice,
                'n_channels': n_channels,
                'n_classes': n_classes
            }, output_path + '/best_unet_model.pth')
            print(f"Saved best model with Dice: {val_dice:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping:
                print(f'Early stopping triggered after {epoch+1} epochs!')
                early_stop = True
                
    model_to_save = load_model(output_path + "/best_unet_model.pth", n_channels, n_classes, unet_depth, device='cpu')
    traced_model = create_traced_model(model_to_save, n_channels, image_size)
    torch.jit.save(traced_model, f"{output_path}/{classifier_name}.pt")
    create_json(classifier_name, f"{output_path}/{classifier_name}.pt", image_size, n_channels, channel_def, pixel_size, class_labels, output_path)

    # Generate plots
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Dice score plot
    plt.subplot(1, 2, 2)
    plt.plot(train_dice_scores, label='Train Dice')
    plt.plot(val_dice_scores, label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Training and Validation Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path + '/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()


# ====================== Prediction Function ======================
def predict(model, image_path, n_channels=3, image_size=256, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Load and preprocess image
        image = tifffile.imread(image_path)
        if len(image.shape)>2:
            image = np.transpose(image, (1, 2, 0)) 

        # Convert to float32 and normalize
        image = min_max_normalization(image)
        
        # Handle channels
        if len(image.shape) == 2:
            image = np.stack([image] * n_channels, axis=-1)
        elif image.shape[-1] != n_channels:
            if image.shape[-1] < n_channels:
                pad = np.zeros((image.shape[0], image.shape[1], n_channels - image.shape[-1]))
                image = np.concatenate([image, pad], axis=-1)
            else:
                image = image[..., :n_channels]

        # Resize if needed (assuming 256x256)
        if image.shape[0] != image_size or image.shape[1] != image_size:
            image = resize(image, (image_size, image_size), preserve_range=True)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

        return probabilities.cpu().numpy()

