"""
Created on Fri April 18 16:30:20 2025

@author: KD264511
"""
import numpy as np
import math
import timeit
import json
from pathlib import Path
import skimage.io as ski
from tensorflow import keras
from keras.utils import normalize  
from Training_lib import Main_model

"=============================================================================================================================================================="
def Ensemble_prediction(X_test, n_classes =4, Patch_size=256, Num_of_models=10):
    # Predict for each model
    stacked = []
    
    for i in range(Num_of_models):

        Save_name      = 'UNet_Norm_256px_1.5kV_'+str(i)
        model_folder   = 'Ensemble_models/'
        history_folder = 'Ensemble_history/'

        model, history = Main_model(Save_name=Save_name, model_folder=model_folder, history_folder=history_folder, 
                                    n_classes=n_classes, Patch_size=Patch_size,
                                    Training_param=False)
    
    
        y_pred_test = model.predict(X_test, verbose=0)
        stacked.append(y_pred_test)
    
    stacked_array  = np.stack(stacked, axis=0) 
    avg_prediction = np.mean(stacked_array, axis=0)
    avg_y_pred_argmax_test = np.argmax(avg_prediction, axis=-1)
    avg_proba_map_pred = np.max(avg_prediction, axis=-1)

    return avg_prediction, avg_y_pred_argmax_test, avg_proba_map_pred

"=============================================================================================================================================================="
def compute_patch_grid_shape(image_shape, patch_size, step=None):
    """
    Compute the grid shape (rows, cols) of patches.

    Parameters:
        image_shape : tuple (H, W) — shape of the original image
        patch_size  : int or tuple (patch_height, patch_width)
        step        : step size (int or tuple), default is equal to patch_size

    Returns:
        (rows, cols) — number of patches along height and width
    """
    H, W = image_shape
    if isinstance(patch_size, int):
        ph, pw = patch_size, patch_size
    else:
        ph, pw = patch_size

    if step is None:
        sh, sw = ph, pw
    elif isinstance(step, int):
        sh, sw = step, step
    else:
        sh, sw = step

    rows = math.floor((H - ph) / sh) + 1
    cols = math.floor((W - pw) / sw) + 1

    return rows, cols
    
"=============================================================================================================================================================="
def unpatchify(patches, grid_shape, patch_size):
    """
    Reconstruct full image from flat patches.
    
    Parameters:
        patches     : (n_patches, patch_size, patch_size)
        grid_shape  : (rows, cols) = shape of patch grid
        patch_size  : int = height and width of each patch
    
    Returns:
        Reconstructed full image (H, W)
    """
    a, b = grid_shape
    full_img = np.zeros((a * patch_size, b * patch_size), dtype=patches.dtype)

    idx = 0
    for i in range(a):
        for j in range(b):
            full_img[i * patch_size:(i + 1) * patch_size,
                     j * patch_size:(j + 1) * patch_size] = patches[idx]
            idx += 1
    return full_img

"=============================================================================================================================================================="
def sliding_patchify_with_canvas(image, px=256, py=256, sx=0, sy=0):
    """
    Splits the image into patches using a sliding window approach with stride,
    and generates for each patch a binary canvas mask of the same size as the input image.
    
    Parameters:
        image (np.ndarray): Input image array of shape (H, W) or (H, W, C)
        px (int): Patch width
        py (int): Patch height
        sx (int): Stride along x-axis (horizontal)
        sy (int): Stride along y-axis (vertical)

    Returns:
        List of tuples: (patch, (x, y), (px, py), canvas_mask)
    """
    H, W = image.shape[:2]
    patches_with_canvas = []
    patch_info = []
    patch_canvas = []

    for y in range(0, H, py):
        for x in range(0, W, px):
            patch = image[y+sy:y+py+sy, x+sx:x+px+sx]
            canvas = np.zeros((H, W), dtype=np.uint8)
            canvas[y+sy:y+py+sy, x+sx:x+px+sx] = 1
            if patch.shape[0] == px and patch.shape[1] == py:
                patches_with_canvas.append(patch)
                patch_info.append((x+sx, y+sy))
                patch_canvas.append(canvas)

    return patches_with_canvas, patch_info, patch_canvas

"=============================================================================================================================================================="
def restitch_from_patches(stacked_patches, positions, image_shape):
    """
    Reconstruct the full image from stacked patches and their top-left positions.

    Parameters:
        stacked_patches (np.ndarray): Array of shape (N, H_p, W_p) or (N, H_p, W_p, C)
        positions (List[Tuple[int, int]]): List of (x, y) top-left positions for each patch
        image_shape (Tuple[int, int] or Tuple[int, int, int]): Shape of full image

    Returns:
        np.ndarray: Reconstructed full image of shape image_shape
    """
    full_image = np.zeros(image_shape, dtype=stacked_patches.dtype)
    
    for patch, (x, y) in zip(stacked_patches, positions):
        h, w = patch.shape[:2]
        full_image[y:y+h, x:x+w] = patch
    
    return full_image

    "=============================================================================================================================================================="
def generate_stride_offsets(patch_size=256, stride=64):
    """
    Generate a list of all (sx, sy) stride offsets to cover a full patch with a given stride,
    avoiding redundant extractions.

    Parameters:
        patch_size (int): Size of the patch (assumes square patches)
        stride (int): Stride value for sliding window

    Returns:
        List[Tuple[int, int]]: List of (sx, sy) tuples
    """
    offsets = []
    for sx in range(0, patch_size, stride):
        for sy in range(0, patch_size, stride):
            offsets.append((sx, sy))
    return offsets

"=============================================================================================================================================================="
def Large_pred(Image_name, Label_name, model = None, Patch_size=256, stride=64, Norm=True, tech=None):

    """
    Perform large-scale prediction on an image using patch-wise prediction and stride shifting (sliding windows).

    Parameters:
        Image_name (str or Path): Path to the input image file to be processed.
        Label_name (str or Path): Path to the corresponding label file.
        model (keras.Model or None): Trained Keras model to use for prediction. Required if tech is not 'Ensemble'.
                                     If tech='Ensemble', this parameter is ignored and multiple models are used.
        Patch_size (int): Size of the square patches to be extracted from the image (default is 256).
        stride (int): Stride value used for sliding window patch extraction (default is 64).
        Norm (bool): If True, normalization is applied to the input patches (default is True).
        tech (str or None): Optional identifier to specify the prediction method or technique used 
                            (e.g., direct U-Net segmentation or ensemble strategies).

    Returns:
        Reconstructed_avg_proba (np.ndarray): Softmax probability map of shape (H, W, num_classes), averaged over all stride offsets.
        final_pred (np.ndarray): Final segmentation map of shape (H, W), containing the class index with the highest probability per pixel.
    """

    s = stride
    start = timeit.default_timer()
    
    print("Processing image :", Image_name)
    
    I = ski.imread(Image_name)
    L = ski.imread(Label_name)

    # Crop image to be dividable py the patch_size
    x_c = int(np.floor(I.shape[0]/Patch_size))
    y_c = int(np.floor(I.shape[1]/Patch_size))
    I_c = I[:x_c*Patch_size, :y_c*Patch_size]
    L_c = L[:x_c*Patch_size, :y_c*Patch_size]
    
    # --- Define parameters ---
    stride_offsets = generate_stride_offsets(patch_size=Patch_size, stride=s)
    image_shape = (I_c.shape[0], I_c.shape[1], 4)
    
    # --- Prepare accumulators ---
    probability_accumulator = np.zeros(image_shape, dtype=np.float32)
    canvas_accumulator = np.zeros(image_shape[:2], dtype=np.float32)
    
    # --- Loop over each stride ---
    c = 0
    for sx, sy in stride_offsets:
        c = c+1
        print('Iteration',c,'out of',len(stride_offsets))
        patches, patches_info, patches_canvas = sliding_patchify_with_canvas(
            I_c, px=Patch_size, py=Patch_size, sx=sx, sy=sy
        )
        
        stack_patches = np.stack(patches, axis=0)
        stack_patches_canvas = np.stack(patches_canvas, axis=0)
        
        images = np.expand_dims(stack_patches, axis=-1)
        
        if Norm:
            test_images = normalize(images, axis=1)
        else:
            test_images = images
    
        if tech == 'Ensemble':
            # Run ensemble prediction
            avg_prediction, _, _ = Ensemble_prediction(
                test_images, n_classes=4, Patch_size=Patch_size, Num_of_models=10
            )
    
        else: 
            # Direct prediction using U-Net:
            if model is None:
                raise ValueError("Technique requires model.")
                
            avg_prediction = model.predict(test_images, verbose=0)
    
        # Reconstruct the full image prediction
        reconstructed_image = restitch_from_patches(avg_prediction, patches_info, image_shape)
        probability_accumulator += reconstructed_image
    
        # Accumulate the canvas (number of contributions per pixel)
        canvas_accumulator += np.sum(stack_patches_canvas, axis=0)
    
    # --- Normalize the final probabilities ---
    canvas_accumulator = np.maximum(canvas_accumulator, 1e-9)  # Prevent division by zero
    Reconstructed_avg_proba = probability_accumulator / canvas_accumulator[..., np.newaxis]
    
    # --- Final class prediction ---
    final_pred = np.argmax(Reconstructed_avg_proba, axis=-1)
    
    
    stop = timeit.default_timer()
    total_time = stop - start
    print(
        f"Patch prediction done. Prediction time is: {np.floor(total_time/60):.0f}:"
        f"{int(total_time - np.floor(total_time/60)*60):02d} min"
    )
    
    return Reconstructed_avg_proba, final_pred

"=============================================================================================================================================================="
def updated_Large_pred(Image_name, Label_name, model = None, Patch_size=256, stride=64, Norm=True, tech=None):

    """
    Perform large-scale prediction on an image using patch-wise prediction and stride shifting (sliding windows).

    Parameters:
        Image_name (str or Path): Path to the input image file to be processed.
        Label_name (str or Path): Path to the corresponding label file.
        model (keras.Model or None): Trained Keras model to use for prediction. Required if tech is not 'Ensemble'.
                                     If tech='Ensemble', this parameter is ignored and multiple models are used.
        Patch_size (int): Size of the square patches to be extracted from the image (default is 256).
        stride (int): Stride value used for sliding window patch extraction (default is 64).
        Norm (bool): If True, normalization is applied to the input patches (default is True).
        tech (str or None): Optional identifier to specify the prediction method or technique used 
                            (e.g., direct U-Net segmentation or ensemble strategies).

    Returns:
        Reconstructed_avg_proba (np.ndarray): Softmax probability map of shape (H, W, num_classes), averaged over all stride offsets.
        final_pred (np.ndarray): Final segmentation map of shape (H, W), containing the class index with the highest probability per pixel.
    """

    s = stride
    start = timeit.default_timer()
    
    print("Processing image :", Image_name)
    
    I = ski.imread(Image_name)
    I_size = I.shape
    
    flip_horiz = np.fliplr(I)
    flip_vert = np.flipud(I)
    flip_both = np.flip(I)
    top = np.concatenate((I, flip_horiz), axis=1)      
    bottom = np.concatenate((flip_vert, flip_both), axis=1)  
    I_new =  np.concatenate((top, bottom), axis=0)    

    I = I_new

    L = ski.imread(Label_name)

    # Crop image to be dividable py the patch_size
    x_c = int(np.floor(I.shape[0]/Patch_size))
    y_c = int(np.floor(I.shape[1]/Patch_size))
    I_c = I[:x_c*Patch_size, :y_c*Patch_size]
    L_c = L[:x_c*Patch_size, :y_c*Patch_size]
    
    # --- Define parameters ---
    stride_offsets = generate_stride_offsets(patch_size=Patch_size, stride=s)
    image_shape = (I_c.shape[0], I_c.shape[1], 4)
    
    # --- Prepare accumulators ---
    probability_accumulator = np.zeros(image_shape, dtype=np.float32)
    canvas_accumulator = np.zeros(image_shape[:2], dtype=np.float32)
    
    # --- Loop over each stride ---
    c = 0
    for sx, sy in stride_offsets:
        c = c+1
        print('Iteration',c,'out of',len(stride_offsets))
        patches, patches_info, patches_canvas = sliding_patchify_with_canvas(
            I_c, px=Patch_size, py=Patch_size, sx=sx, sy=sy
        )
        
        stack_patches = np.stack(patches, axis=0)
        stack_patches_canvas = np.stack(patches_canvas, axis=0)
        
        images = np.expand_dims(stack_patches, axis=-1)
        
        if Norm:
            test_images = normalize(images, axis=1)
        else:
            test_images = images
    
        if tech == 'Ensemble':
            # Run ensemble prediction
            avg_prediction, _, _ = Ensemble_prediction(
                test_images, n_classes=4, Patch_size=Patch_size, Num_of_models=10
            )
    
        else: 
            # Direct prediction using U-Net:
            if model is None:
                raise ValueError("Technique requires model.")
                
            avg_prediction = model.predict(test_images, verbose=0)
    
        # Reconstruct the full image prediction
        reconstructed_image = restitch_from_patches(avg_prediction, patches_info, image_shape)
        probability_accumulator += reconstructed_image
    
        # Accumulate the canvas (number of contributions per pixel)
        canvas_accumulator += np.sum(stack_patches_canvas, axis=0)
    
    # --- Normalize the final probabilities ---
    canvas_accumulator = np.maximum(canvas_accumulator, 1e-9)  # Prevent division by zero
    Reconstructed_avg_proba = probability_accumulator / canvas_accumulator[..., np.newaxis]
    
    # --- Final class prediction ---
    final_pred = np.argmax(Reconstructed_avg_proba, axis=-1)
    final_pred = final_pred[0:I_size[0], 0:I_size[1]]
    
    
    stop = timeit.default_timer()
    total_time = stop - start
    print(
        f"Patch prediction done. Prediction time is: {np.floor(total_time/60):.0f}:"
        f"{int(total_time - np.floor(total_time/60)*60):02d} min"
    )
    
    return Reconstructed_avg_proba, final_pred

"=============================================================================================================================================================="
