"""
Created on Wed May 29 15:11:19 2024

@author: KD264511
"""
import numpy as np
from patchify import patchify
import random
import albumentations as A

"=============================================================================================================================================================="
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1 (tuple): Bounding box 1 (x1, y1, w1, h1)
    box2 (tuple): Bounding box 2 (x2, y2, w2, h2)
    
    Returns:
    float: The IoU of the two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    # Calculate the area of the intersection rectangle
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_box1 = w1 * h1
        area_box2 = w2 * h2
        union = area_box1 + area_box2 - intersection
        return intersection / union
    
    return 0.0
    
"=============================================================================================================================================================="
def generate_boxes(image_width, image_height, num_boxes, iou_threshold, patch_width, patch_height):
    """
    Generate random bounding boxes with an IoU constraint.
    
    Parameters:
    image_width (int): The width of the image.
    image_height (int): The height of the image.
    num_boxes (int): Number of patches to generate.
    iou_threshold (float): Intersection over Union threshold for patches.
    patch_width (int): Width of the patches.
    patch_height (int): Height of the patches.
    
    Returns:
    list: A list of generated boxes.
    """
    boxes = []
    max_attempts = num_boxes * 1000
    attempts = 0
    
    while len(boxes) < num_boxes and attempts < max_attempts:
        attempts += 1
        # Generate a random bounding box
        width = patch_width # np.random.randint(10, image_width // 2)
        height = patch_height # np.random.randint(10, image_height // 2)
        x = np.random.randint(0, image_width - width)
        y = np.random.randint(0, image_height - height)
        new_box = (x, y, width, height)
        
        # Check the IoU with existing boxes
        iou_checks = [calculate_iou(new_box, box) for box in boxes]
        if all(iou < iou_threshold for iou in iou_checks):
            boxes.append(new_box)
    
    return boxes

"=============================================================================================================================================================="
def generate_patches(large_image, large_label, patch_size, Type='Seq', Num_patch=None, iou_threshold=None):
    """
    Generates patches from a large image and corresponding label.

    Parameters:
    large_image (numpy.ndarray): The large image to be patched.
    large_label (numpy.ndarray): The large label to be patched.
    patch_size (int): The size of the patches to generate.
    Type (str): The type of patch generation ('Seq' for sequential, 'Rand' for random).
    Num_patch (int, optional): Maximum number of generated patches (only for 'Rand').
    iou_threshold (float, optional): IoU threshold (only for 'Rand').

    Returns:
    tuple: Patches of images and labels.
    """
    if Type == 'Seq':
        try:
            patches_img = patchify(large_image, (patch_size, patch_size), step=patch_size)
            a, b, sizex, sizey = patches_img.shape
            patches_img = patches_img.reshape(a*b, sizex, sizey)

            patches_lbl = patchify(large_label, (patch_size, patch_size), step=patch_size)
            patches_lbl = patches_lbl.reshape(a*b, sizex, sizey)

            return patches_img, patches_lbl
        except Exception as e:
            raise ValueError(f"Error generating sequential patches: {e}")

    elif Type == 'Rand':
        if Num_patch is None or iou_threshold is None:
            raise ValueError("Num_patch and iou_threshold must be provided when Type is 'Rand'.")

        try:
            image_width, image_height = large_image.shape[0], large_image.shape[1]
            generated_boxes = generate_boxes(image_width, image_height, Num_patch, iou_threshold, patch_size, patch_size)

            patches_img = []
            patches_lbl = []
            for box in generated_boxes:
                x, y, w, h = box
                patches_img.append(large_image[x:x+w, y:y+h])
                patches_lbl.append(large_label[x:x+w, y:y+h])

            patches_img = np.array(patches_img)
            patches_lbl = np.array(patches_lbl)

            return patches_img, patches_lbl
        except Exception as e:
            raise ValueError(f"Error generating random patches: {e}")

    else:
        raise ValueError("Unsupported patch generation type. Use 'Seq' for sequential or 'Rand' for random.")

"=============================================================================================================================================================="
def Image_augmentation(image, mask, num_transformations=5):
    def generate_transform(seed=None):
        # Optionally set a random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Define the transformation pipeline
        transform = A.Compose([
            # Geometric transforms
            A.Flip(p=0.5),  # Flip the image vertically or horizontally
            A.HorizontalFlip(p=0.5),  # Flip the image horizontally, around the y-axis
            A.Rotate(limit=45, p=0.5),  # Rotate the image by a randomly selected angle within the given limit
            # Color transforms
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2, 
                                       p=0.5),  # Randomly adjust brightness and contrast
            # Blur transforms
            A.GaussianBlur(blur_limit=3, p=0.5),  # Apply Gaussian blur
            # Noise transforms
            A.GaussNoise(p=0.5),  # Apply Gaussian noise
            # Additional transforms for more variability
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ])
        return transform

    transformed_images = []
    transformed_masks = []

    for i in range(num_transformations):
        seed = np.random.randint(0, 10000)  # Generate a new random seed for each transformation
        transform = generate_transform(seed)

        # Apply the transformations
        transformed = transform(image=image, masks=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['masks']
        
        
        # transformed = transform(image=image, masks=mask)
        # transformed_image1 = transformed['image']
        # transformed_masks  = transformed['masks']

        transformed_image2 = transformed_mask[0] 
        transformed_mask = transformed_mask[1] 
        
        transformed_image = np.stack((transformed_image, transformed_image2,), axis=-1)
        transformed_image = np.transpose(transformed_image, (2, 0, 1))
        
        # Collect the transformed images and masks
        transformed_images.append(transformed_image)
        transformed_masks.append(transformed_mask)
    
    return transformed_images, transformed_masks

"=============================================================================================================================================================="

