import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def readImageData(rootpath):
    '''Reads data 
    Arguments: path to the image, for example './Training'
    Returns:   list of images, list of corresponding outputs'''
    images = [] # images
    output_1 = [] # corresponding x index
    output_2 = [] # corresponding y index
    output_3 = [] # corresponding x width
    output_4 = [] # corresponding y width
    
    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader)
    # loop over all images in current annotations file
    for row in gtReader:
        img=Image.open(prefix + row[0])  # the 1th column is the filename
        # preprocesing image, here we resize the image into a smaller one
        img=img.resize((128,128), Image.BICUBIC)  
        img=np.array(img)
        images.append(img) 
        output_1.append(float(row[1])) # the 8th column is the label
        output_2.append(float(row[2]))
        output_3.append(float(row[3]))
        output_4.append(float(row[4]))
    
    gtFile.close()
    return images, output_1, output_2, output_3, output_4

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def preprocess_images(images):
    """
    Preprocess a list of images for VGG.
    Args:
        images: List of images (NumPy arrays)
    Returns:
        torch.Tensor: Preprocessed images as a 4D tensor
    """
    # Define ImageNet normalization
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(224, padding=4),  # 随机裁剪并填充
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    # Apply transformation to each image
    processed_images = [transform(img) for img in images]
    
    # Stack images into a single tensor (batch)
    return torch.stack(processed_images)

def visualize_images(images, labels=None, num_images=5):
    """
    Visualize a few images with optional labels.
    
    Args:
        images (torch.Tensor): A batch of images with shape (N, C, H, W).
        labels (torch.Tensor or list, optional): Labels corresponding to the images. Defaults to None.
        num_images (int): Number of images to visualize. Defaults to 5.
    """
    # Ensure images are on the CPU and converted to NumPy
    images = images.cpu().numpy()
    
    # Normalize and reshape images for display
    images = np.transpose(images, (0, 2, 3, 1))  # Convert from (N, C, H, W) to (N, H, W, C)
    
    # Clip values to valid range [0, 1] if needed
    if images.max() > 1.0:
        images = images / 255.0
    
    # Display a few images
    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        if labels is not None:
            plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_images(images, rows, cols, titles=None):
    """
    Display a group of images in a grid.
    Arguments:
        images: List of images (each as a NumPy array).
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        titles: Optional list of titles for each image.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the grid for easy iteration
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')  # Display the image
            ax.axis('off')  # Hide axes
            if titles:
                ax.set_title(titles[i], fontsize=10)
        else:
            ax.axis('off')  # Hide extra axes
    plt.tight_layout()
    plt.show()

# Function to calculate the error between predicted and ground truth values
def calculate_error(pred, gt):
    # Calculate the Euclidean distance between predicted and ground truth (x, y, xw, yw)
    pred_x, pred_y, pred_xw, pred_yw = pred
    gt_x, gt_y, gt_xw, gt_yw = gt
    
    # Calculate absolute errors for each dimension
    error_x = abs(pred_x - gt_x)
    error_y = abs(pred_y - gt_y)
    error_xw = abs(pred_xw - gt_xw)
    error_yw = abs(pred_yw - gt_yw)
    
    # Sum of absolute errors (you can also use squared error or Euclidean distance)
    total_error = error_x + error_y + error_xw + error_yw
    return total_error

def draw_rectangle(ax, center_x, center_y, x_width, y_width, color, scale_factor_x, scale_factor_y):
    """
    Draw a rectangle on the image.
    Arguments:
    - ax: Matplotlib axis object to draw on
    - center_x, center_y: Center coordinates of the rectangle
    - x_width, y_width: Width and height of the rectangle
    - color: Rectangle border color
    - scale_factor: Scaling factor for resizing coordinates
    """
    # Scale the coordinates and dimensions
    center_x_scaled = center_x * scale_factor_x
    center_y_scaled = center_y * scale_factor_y
    x_width_scaled = x_width * scale_factor_x
    y_width_scaled = y_width * scale_factor_y

    # Calculate the top-left corner of the rectangle
    top_left_x = center_x_scaled - x_width_scaled / 2
    top_left_y = center_y_scaled - y_width_scaled / 2

    # Create a rectangle patch
    rectangle = patches.Rectangle(
        (top_left_x, top_left_y),  # Top-left corner
        x_width_scaled,           # Width
        y_width_scaled,           # Height
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rectangle)


