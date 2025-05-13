import numpy as np
from keras.utils import normalize
# import glob
import random
from tqdm import tqdm 
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, normalize
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout#, BatchNormalization, UpSampling2D,
from sklearn.preprocessing import LabelEncoder

import json
import skimage as ski
import keras
import re

"=============================================================================================================================================================="
def natural_sort_key(path):
    """Extract numerical values from filenames for natural sorting."""
    return [int(num) if num.isdigit() else num for num in re.findall(r'\d+', str(path))]

def Load_training_data(image_paths, mask_paths, SIZE_X, SIZE_Y, SIZE_Z=1, Total_num_images=100, shuffle=False):
    """
    Load and preprocess training images and masks.

    Parameters:
    image_paths (list)    : List of paths to the image files.
    mask_paths (list)     : List of paths to the mask files.
    SIZE_X (int)          : Target width of images and masks.
    SIZE_Y (int)          : Target height of images and masks.
    Total_num_images (int): Total number of images used for training, validation, and testing.
    shuffle (bool)        : If True, shuffle the order of images and masks.

    Returns:
    tuple                 : Tuple containing arrays of processed images and masks.
    """
    
    # Sort the image and mask paths in natural order
    image_paths = sorted(image_paths, key=natural_sort_key)
    mask_paths = sorted(mask_paths, key=natural_sort_key)

    # Limit to Total_num_images
    image_paths = image_paths[:Total_num_images]
    mask_paths = mask_paths[:Total_num_images]

    if shuffle:
        combined = list(zip(image_paths, mask_paths))
        random.shuffle(combined)
        image_paths, mask_paths = zip(*combined)

    "==========================================================================================================================="
    " Load images =============================================================================================================="
    "==========================================================================================================================="
    train_images = []
    print("Loading images...")
    for img_path in tqdm(image_paths, total=len(image_paths)):
        img = cv2.imread(str(img_path), 0)
        if img is None:
            print(f"Failed to load image at {img_path}")
            continue
        img = cv2.resize(img, (SIZE_X, SIZE_Y))  # Resize image
        train_images.append(img)

    train_images = np.array(train_images, dtype=np.uint8)
    print("Images loaded and processed. Shape:", train_images.shape)

    "==========================================================================================================================="
    " Load masks ==============================================================================================================="
    "==========================================================================================================================="
    train_masks = []
    print("Loading masks...")
    for msk_path in tqdm(mask_paths, total=len(mask_paths)):
        mask = cv2.imread(str(msk_path), 0)
        if mask is None:
            print(f"Failed to load mask at {msk_path}")
            continue
        mask = cv2.resize(mask, (SIZE_X, SIZE_Y))  # Resize mask
        train_masks.append(mask)

    train_masks = np.array(train_masks, dtype=np.uint8)
    
    print("Masks loaded and processed. Shape:", train_masks.shape)

    return train_images, train_masks
    
"=============================================================================================================================================================="
def Split_data(train_images, train_masks, perc = 0.8, val_perc = 0.2, n_classes = 4):
    """
    Split data into training, validation, and testing datasets 

    Parameters:
    train_images (np.array): Array of training images.
    train_masks (np.array) : Array of training masks.
    perc (float)           : Percentage of data to use for training and validation.
    val_perc (float)       : Percentage of training data to use for validation.
    n_classes (int)        : Number of classes for segmentation.

    Returns:
    tuple: Tuple containing training, validation, and test datasets.
    """

    # Check the shape of the mask array to determine if we need to expand dimensions 
    if len(train_masks.shape) == 3:
        # Expand the last dimension of mask if it doesn't have channel dimension
        train_masks = np.expand_dims(train_masks, axis = 3)
        
    elif len(train_masks.shape) == 4:
        train_masks = train_masks
        
    else:
        # Raise an error if the mask shape is not 3D or 4D
        raise ValueError("Inconsistency in the training mask shape; make sure that the length of the shape is equal to 3 or 4.")

    if perc == 1:
        X_test  = train_images
        y_test  = train_masks
        X_train = []
        y_train = [] 
        X_val   = []
        y_val   = []
        
        # Convert labels to one-hot encoding format for use in categorical crossentropy 
        y_train_cat = []
        y_val_cat   = []
        y_test_cat  = to_categorical(y_test,  num_classes=n_classes).reshape((y_test.shape[0],  y_test.shape[1],  y_test.shape[2],  n_classes))
        
        # Output the counts of each dataset
        print("Number of Training samples    = 0")
        print("Number of Validation samples  = 0")
        print("Number of Test samples        =", X_test.shape[0])
        
    elif perc == 0:
        # If perc == 0, use all data for training and validation, no test set
        X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, 
                                                            test_size = val_perc,
                                                            shuffle = False)
        X_test = []
        y_test = []  
        
        # Convert labels to one-hot encoding format for use in categorical crossentropy 
        y_train_cat = to_categorical(y_train, num_classes=n_classes).reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
        y_val_cat   = to_categorical(y_val,   num_classes=n_classes).reshape((y_val.shape[0],   y_val.shape[1],   y_val.shape[2],   n_classes))
        y_test_cat  = []
        
        # Output the counts of each dataset
        print("Number of Training samples    =", X_train.shape[0])
        print("Number of Validation samples  =", X_val.shape[0])
        print("Number of Test samples        = 0")
        
    elif 0 < perc < 1:
        # Split the data into training and test subsets
        X_train_1, X_test, y_train_1, y_test = train_test_split(train_images, train_masks, 
                                                                test_size = 1-perc,
                                                                shuffle = False)
            
        # Further split the training data into training and validation subsets 
        X_train, X_val, y_train, y_val = train_test_split(X_train_1, y_train_1,
                                                            test_size = val_perc,
                                                            shuffle = False)
            
        # Convert labels to one-hot encoding format for use in categorical crossentropy 
        y_train_cat = to_categorical(y_train, num_classes=n_classes).reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
        y_val_cat   = to_categorical(y_val,   num_classes=n_classes).reshape((y_val.shape[0],   y_val.shape[1],   y_val.shape[2],   n_classes))
        y_test_cat  = to_categorical(y_test,  num_classes=n_classes).reshape((y_test.shape[0],  y_test.shape[1],  y_test.shape[2],  n_classes))
    
        # Output the counts of each dataset
        print("Number of Training samples    = ", X_train.shape[0])
        print("Number of Validation samples  = ", X_val.shape[0]  )
        print("Number of Test samples        = ", X_test.shape[0] )

    return X_train, y_train, y_train_cat, X_val, y_val, y_val_cat, X_test, y_test, y_test_cat

"=============================================================================================================================================================="
def Load_and_prepare_data(for_training, data_dir, patch_size, val_perc = 0.2, n_classes=4, normalize_images=True, shuffle=True, nb_samples = None):

    
    """Load, encode, and split training data for segmentation."""

    if for_training == True:   

        image_dir = data_dir[0]
        label_dir = data_dir[1]
    
        image_paths = sorted(image_dir.glob('*.tif'))
        label_paths = sorted(label_dir.glob('*.tif'))

        if nb_samples == None:
            num_samples = len(image_paths)
        else:
            num_samples = nb_samples
    
        # Load image and mask arrays
        images, masks = Load_training_data(
            image_paths, label_paths,
            SIZE_X=patch_size, SIZE_Y=patch_size, SIZE_Z=1,
            Total_num_images=num_samples,
            shuffle=shuffle
        )
    
        # Encode labels
        label_encoder = LabelEncoder()
        masks_flat = masks.reshape(-1)
        masks_encoded = label_encoder.fit_transform(masks_flat)
        masks_encoded = masks_encoded.reshape(masks.shape)
    
        # Expand dims to add channel dimension
        images = np.expand_dims(images, axis=-1)
        masks_encoded = np.expand_dims(masks_encoded, axis=-1)
    
        # Normalize images
        if normalize_images:
            images = normalize(images, axis=1)
    
        # Split into training, validation, and test sets
        X_train, y_train, y_train_cat, X_val, y_val, y_val_cat, X_test, y_test, y_test_cat = Split_data(
            images, masks_encoded, 
            perc=0, val_perc=val_perc, 
            n_classes=n_classes
        )
        
    else:

        image_dir = data_dir[0]
        label_dir = data_dir[1]
    
        image_paths = sorted(image_dir.glob('*.tif'))
        label_paths = sorted(label_dir.glob('*.tif'))
    
        if nb_samples == None:
            num_samples = len(image_paths)
        else:
            num_samples = nb_samples
    
        # Load image and mask arrays
        images, masks = Load_training_data(
            image_paths, label_paths,
            SIZE_X=patch_size, SIZE_Y=patch_size, SIZE_Z=1,
            Total_num_images=num_samples,
            shuffle=shuffle
        )
    
        # Encode labels
        label_encoder = LabelEncoder()
        masks_flat = masks.reshape(-1)
        masks_encoded = label_encoder.fit_transform(masks_flat)
        masks_encoded = masks_encoded.reshape(masks.shape)
    
        # Expand dims to add channel dimension
        images = np.expand_dims(images, axis=-1)
        masks_encoded = np.expand_dims(masks_encoded, axis=-1)
    
        # Normalize images
        if normalize_images:
            images = normalize(images, axis=1)
    
        # Split into training, validation, and test sets
        X_train, y_train, y_train_cat, X_val, y_val, y_val_cat, X_test, y_test, y_test_cat = Split_data(
            images, masks_encoded, 
            perc=1, val_perc=0, 
            n_classes=n_classes
        )
        

    return X_train, y_train, y_train_cat, X_val, y_val, y_val_cat, X_test, y_test, y_test_cat

"=============================================================================================================================================================="
def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    """
    Build a U-Net model for semantic segmentation 
    
    Parameters:
    n_classes (int)   : Number of output classes.
    IMG_HEIGHT (int)  : Height of the input image.
    IMG_WIDTH (int)   : Width of the input image.
    IMG_CHANNELS (int): Number of image channels.

    Returns:
    Model             : A Model object containing the U-Net model.
    """
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs  # Input normalization can be done outside if required
    
    # Contraction path 
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1, training=True)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2, training=True)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3, training=True)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4, training=True)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5, training=True)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Expansive path    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6, training=True)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7, training=True)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8, training=True)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9, training=True)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    # Output Layer     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])

    return model

"=============================================================================================================================================================="

def Main_model(X_train=None, y_train_cat=None, X_val=None, y_val_cat=None,
               Save_name=None, model_folder=None, history_folder=None, n_classes=4, Patch_size = 256,
               Training_param = True, Summary_param = False, 
               batch_size = 32, epochs = 100, learning_rate = 1e-3):

    """
    Apply augmentation transformations to an image and its corresponding mask and return the transformed images.

    Parameters:
    X_train (array)      : Training data.
    y_train_cat (array)  : One-hot encoded training masks.
    X_val (array)        : Validation data.
    y_val_cat (array)    : One-hot encoded validation masks.
    Save_name (str)      : Name to save/load the model file.
    model_folder (str)   : Folder to save/load the model.
    history_folder (str) : Folder to save/load the training history.
    Training_param (bool): If True, trains a new model; if False, loads pretrained.
    Summary_param (bool) : If True, prints model summary.
    batch_size (int)     : Training batch size.
    epochs (int)         : Number of training epochs.
    learning_rate (float): Learning rate.
    
    
    Returns:
    model    : Trained or loaded Keras model.
    history  : Training history object or wrapper.
    """
    


    if Training_param:
        if any(v is None for v in [X_train, y_train_cat, X_val, y_val_cat]):
            raise ValueError("Training_param=True requires X_train, y_train_cat, X_val, and y_val_cat.")
            
        IMG_HEIGHT   = X_train.shape[1]
        IMG_WIDTH    = X_train.shape[2]
        IMG_CHANNELS = X_train.shape[3]

    else:
        # For test-only mode, get image dimensions from saved model later if needed
        # Placeholder dimensions for model instantiation (can be adjusted)
        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = Patch_size, Patch_size, 1  # Default/fallback values
         # --- Define model ---
        model = multi_unet_model(n_classes=n_classes, 
                                 IMG_HEIGHT=IMG_HEIGHT, 
                                 IMG_WIDTH=IMG_WIDTH, 
                                 IMG_CHANNELS=IMG_CHANNELS)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
        if Summary_param:
            model.summary()
    
    if not Training_param: # Load model 
        
        # Define model
        model = multi_unet_model(n_classes    = n_classes, 
                                 IMG_HEIGHT   = IMG_HEIGHT, 
                                 IMG_WIDTH    = IMG_WIDTH, 
                                 IMG_CHANNELS = IMG_CHANNELS)
        model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Print model summary
        if Summary_param:
            model.summary()
            
        # Load the trained weights to the defined model 
        model.load_weights(model_folder + Save_name + '.hdf5')

        # Load model history
        file_name = history_folder + 'History_' + Save_name + '.json'
        try:
            with open(file_name, 'r') as f:
                content = f.read()
                if not content:
                    print("Error: The file is empty.")
                else:
                    try:
                        history = json.loads(content)
                        print("History loaded successfully")
                    except json.JSONDecodeError as e:
                        print("JSONDecodeError:", e)
        except FileNotFoundError:
            print(f"Error: The file {file_name} does not exist.")
        except Exception as e:
            print("An unexpected error occurred:", e)

        # Create a Simple Wrapper Class:
        class HistoryWrapper:
            def __init__(self, history_dict):
                self.history = history_dict

        history = HistoryWrapper(history)

    else:
        # --- Train the model ---
        history = model.fit(X_train, y_train_cat, 
                            batch_size=batch_size, 
                            verbose=1, 
                            epochs=epochs, 
                            validation_data=(X_val, y_val_cat))

        # Save model and training history
        model.save(model_folder + Save_name + '.hdf5')
        print(f"Model saved to {model_folder + Save_name + '.hdf5'}")

        file_path = history_folder + 'History_' + Save_name + '.json'
        with open(file_path, 'w') as f:
            json.dump(history.history, f)
            print(f"History saved to {file_path}")

    return model, history

"=============================================================================================================================================================="

# def predict_with_uncertainty(model, input_data, n_iter=150):
#     # Define a tf.function to enable training mode during inference
#     #@tf.function
#     def model_predict(x):
#         return model(x, training=True)

#     # Get the model output shape after a single inference run
#     output_shape = model_predict(input_data).shape

#     # Create an array to store results, excluding the batch dimension
#     result = np.zeros((n_iter,) + output_shape[1:])  # (n_iter, 256, 256, 4)

#     for i in range(n_iter):
#         result[i] = model_predict(input_data)

#     prediction_mean = result.mean(axis=0)
#     prediction_std = result.std(axis=0)
#     return result, prediction_mean, prediction_std

# "=============================================================================================================================================================="

# def Prediction_large_uncertainty(IMAGE, Patch_size, Norm, MODEL, n_iter=150):

#     """
#     Segmentation prediction.

#     Parameters:
#     IMAGE (2D array) : Original image.
#     Patch_size (int) : Indicates patch size to decompose image.
#     Norm (bool)      : Normalization parameter, if True the data is normalized.
#     MODEL (str)      : The model used for prediction.
#     n_iter (int)     : Number of iteration where the dropout is varying

#     Returns:
#     tuple: Tuple containing the predicted segmentation of the original image.
#     """

   
#     if len(IMAGE.shape) != 2:
#         raise ValueError("Expected a 2-dimensional array, but received a " + str(len(IMAGE.shape)) + "-dimensional array.")

#     # Split image in four categories:
#     img_height,img_width = IMAGE.shape
    
#     m_h = img_height//Patch_size
#     m_w = img_width//Patch_size
    
#     r_h = img_height%Patch_size
#     r_w = img_width%Patch_size
    
#     # First category:
#     all_patchs = []
#     all_vars = []
#     for j in range(m_h):
#         for k in range(m_w):
#             Patch = IMAGE[Patch_size*j:Patch_size*(j+1), Patch_size*k:Patch_size*(k+1)]
#             single_patch_img = np.expand_dims(Patch, axis=2)  # If grayscale, ensure it's 3D
#             single_patch_img = np.expand_dims(single_patch_img, axis=0)  # Add batch dimension
#             single_patch_img = np.array(single_patch_img, dtype=np.uint8)
#             if Norm == True:
#                 single_patch_img = normalize(single_patch_img, axis=1)
#             # Prediction
#             _, pred_mean, pred_var = predict_with_uncertainty(MODEL, single_patch_img, n_iter)
#             pred = np.argmax(pred_mean, axis=-1)
#             var  = np.max(pred_var, axis=-1)
#             all_patchs.append(pred)
#             all_vars.append(var)
#     # Reassemble the patches into a large image:
#     reconstructed_img = np.zeros((img_height, img_width), dtype=IMAGE.dtype)
#     reconstructed_var = np.zeros((img_height, img_width))
#     patch_index = 0
#     for j in range(m_h):
#         for k in range(m_w):
#             reconstructed_img[Patch_size*j:Patch_size*(j+1), Patch_size*k:Patch_size*(k+1)] = all_patchs[patch_index]
#             reconstructed_var[Patch_size*j:Patch_size*(j+1), Patch_size*k:Patch_size*(k+1)] = all_vars[patch_index]
#             patch_index += 1
            
#     # Second category:
#     all_patchs2 = []
#     all_vars2 = []
#     for k in range(m_w):
#         dd_h = (m_h-1)*Patch_size + r_h
#         Patch1 = IMAGE[dd_h:IMAGE.shape[0], Patch_size*k:Patch_size*(k+1)]
#         single_patch_img = np.expand_dims(Patch1, axis=2)  # If grayscale, ensure it's 3D
#         single_patch_img = np.expand_dims(single_patch_img, axis=0)  # Add batch dimension
#         single_patch_img = np.array(single_patch_img, dtype=np.uint8)
#         if Norm == True:
#             single_patch_img = normalize(single_patch_img, axis=1)
#         # Prediction
#         _, pred_mean, pred_var = predict_with_uncertainty(MODEL, single_patch_img, n_iter)
#         pred = np.argmax(pred_mean, axis=-1)
#         var  = np.mean(pred_var, axis=-1)
#         all_patchs2.append(pred)
#         all_vars2.append(var)
#     reconstructed_img2 = np.zeros((img_height, img_width), dtype=IMAGE.dtype)
#     reconstructed_var2 = np.zeros((img_height, img_width))
#     patch_index = 0 # Assuming all full-size patches are added first in the list
#     for k in range(m_w):
#         dd_h = (m_h-1)*Patch_size + r_h
#         reconstructed_img2[dd_h:IMAGE.shape[0], Patch_size*k:Patch_size*(k+1)] = all_patchs2[patch_index]
#         reconstructed_var2[dd_h:IMAGE.shape[0], Patch_size*k:Patch_size*(k+1)] = all_vars2[patch_index]
#         patch_index += 1
    
#     # Third category:
#     all_patchs3 = []
#     all_vars3 = []
#     for j in range(m_h):
#         dd_w = (m_w-1)*Patch_size + r_w  
#         Patch3 = IMAGE[Patch_size*j:Patch_size*(j+1), dd_w:IMAGE.shape[1]]
#         single_patch_img = np.expand_dims(Patch3, axis=2)  # If grayscale, ensure it's 3D
#         single_patch_img = np.expand_dims(single_patch_img, axis=0)  # Add batch dimension
#         single_patch_img = np.array(single_patch_img, dtype=np.uint8)
#         if Norm == True:
#             single_patch_img = normalize(single_patch_img, axis=1)
#         # Prediction
#         _, pred_mean, pred_var = predict_with_uncertainty(MODEL, single_patch_img, n_iter)
#         pred = np.argmax(pred_mean, axis=-1)
#         # print(pred.shape)
#         var  = np.mean(pred_var, axis=-1)
#         all_patchs3.append(pred)
#         all_vars3.append(var)
#     reconstructed_img3 = np.zeros((img_height, img_width), dtype=IMAGE.dtype)
#     reconstructed_var3 = np.zeros((img_height, img_width))
#     patch_index = 0 
#     for j in range(m_h):
#         dd_w = (m_w-1)*Patch_size + r_w  
#         reconstructed_img3[Patch_size*j:Patch_size*(j+1), dd_w:IMAGE.shape[1]] = all_patchs3[patch_index]
#         reconstructed_var3[Patch_size*j:Patch_size*(j+1), dd_w:IMAGE.shape[1]] = all_vars3[patch_index]
#         patch_index += 1
    
    
#     # Fourth category:
#     Patch4 = IMAGE[dd_h:IMAGE.shape[0], dd_w:IMAGE.shape[1]]
#     single_patch_img = np.expand_dims(Patch4, axis=2)  # If grayscale, ensure it's 3D
#     single_patch_img = np.expand_dims(single_patch_img, axis=0)  # Add batch dimension
#     single_patch_img = np.array(single_patch_img, dtype=np.uint8)
#     if Norm == True:
#         single_patch_img = normalize(single_patch_img, axis=1)
        
#     # Prediction:
#     _, pred_mean, pred_var = predict_with_uncertainty(MODEL, single_patch_img, n_iter)
#     pred = np.argmax(pred_mean, axis=-1)
#     var  = np.mean(pred_var, axis=-1)

    
#     reconstructed_img4 = np.zeros((img_height, img_width), dtype=IMAGE.dtype)
#     reconstructed_img4[dd_h:IMAGE.shape[0], dd_w:IMAGE.shape[1]] = pred 
    
#     reconstructed_var4 = np.zeros((img_height, img_width))
#     reconstructed_var4[dd_h:IMAGE.shape[0], dd_w:IMAGE.shape[1]] = var 

#     # Final image reconstruction: 
#     I1 = np.maximum(reconstructed_img, reconstructed_img2)
#     I2 = np.maximum(I1, reconstructed_img3)
#     I_final = np.maximum(I2, reconstructed_img4)
    
#     I1 = np.maximum(reconstructed_var, reconstructed_var2)
#     I2 = np.maximum(I1,reconstructed_var3)
#     I_var = np.maximum(I2,reconstructed_var4)

#     return I_final, I_var

# "=============================================================================================================================================================================================="
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Activation, Multiply, Lambda
# from tensorflow.keras.models import Model
# import tensorflow.keras.backend as K

# def attention_block(x, g, inter_channel):
#     """
#     Attention gate mechanism for U-Net
    
#     Parameters:
#     x            : Skip connection from the encoder
#     g            : Upsampling layer from the decoder
#     inter_channel: Intermediate channel size for attention computation

#     Returns:
#     Attention-modulated skip connection
#     """
#     theta_x = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(x)
#     phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(g)
    
#     add_xg = Activation('relu')(theta_x + phi_g)
#     psi = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(add_xg)
#     psi = Activation('sigmoid')(psi)
    
#     return Multiply()([x, psi])

# def multi_attention_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#     """
#     Build an Attention U-Net model for semantic segmentation 
    
#     Parameters:
#     n_classes (int)   : Number of output classes.
#     IMG_HEIGHT (int)  : Height of the input image.
#     IMG_WIDTH (int)   : Width of the input image.
#     IMG_CHANNELS (int): Number of image channels.

#     Returns:
#     Model             : A Model object containing the Attention U-Net model.
#     """
    
#     inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     s = inputs
    
#     # Contraction path
#     c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
#     c1 = Dropout(0.1)(c1, training=True)
#     c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)

#     c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#     c2 = Dropout(0.1)(c2, training=True)
#     c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2)

#     c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#     c3 = Dropout(0.2)(c3, training=True)
#     c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     p3 = MaxPooling2D((2, 2))(c3)

#     c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#     c4 = Dropout(0.2)(c4, training=True)
#     c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     p4 = MaxPooling2D(pool_size=(2, 2))(c4)

#     c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#     c5 = Dropout(0.3)(c5, training=True)
#     c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#     # Expansive path with Attention Gates
#     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     attn_6 = attention_block(c4, u6, 128)  # Attention on the skip connection
#     u6 = concatenate([u6, attn_6])
#     c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#     c6 = Dropout(0.2)(c6, training=True)
#     c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#     attn_7 = attention_block(c3, u7, 64)
#     u7 = concatenate([u7, attn_7])
#     c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#     c7 = Dropout(0.2)(c7, training=True)
#     c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

#     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#     attn_8 = attention_block(c2, u8, 32)
#     u8 = concatenate([u8, attn_8])
#     c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#     c8 = Dropout(0.1)(c8, training=True)
#     c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

#     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#     attn_9 = attention_block(c1, u9, 16)
#     u9 = concatenate([u9, attn_9])
#     c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#     c9 = Dropout(0.1)(c9, training=True)
#     c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
#     # Output Layer     
#     outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    
#     model = Model(inputs=[inputs], outputs=[outputs])

#     return model



# def Att_Main_model(X_train, y_train_cat, X_val, y_val_cat,
#                Save_name, model_folder, history_folder, #sample_weights, 
#                Training_param = True, Summary_param = False, 
#                batch_size = 32, epochs = 100, learning_rate = 1e-3):

#     """
#     Apply augmentation transformations to an image and its corresponding mask and return the transformed images.

#     Parameters:
#     X_train (array)      : Training data set.
#     y_train_cat (array)  : Training images masks.
#     X_val (array)        : Validation data set.
#     y_val_cat (array)    : Validation images masks.
#     Training_param (bool): If False then a pretrained model is loaded, if new model is trained.
#     Summary_param (bool) : If True then the model summary is previewed.
#     batch_size (int)     : Batch size for the training process.
#     epochs (int)         : Number of epochs for the training process.
#     learning_rate (float): Learning rate.
    
    
#     Returns:
#     tuple: Tuple containing the splitted images and their masks.
#     """
    
#     IMG_HEIGHT   = X_train.shape[1]
#     IMG_WIDTH    = X_train.shape[2]
#     IMG_CHANNELS = X_train.shape[3]
#     n_classes = 4

    
#     if not Training_param:
#         # Load model 
#         # Define model
#         model = multi_attention_unet_model(n_classes    = n_classes, 
#                                  IMG_HEIGHT   = IMG_HEIGHT, 
#                                  IMG_WIDTH    = IMG_WIDTH, 
#                                  IMG_CHANNELS = IMG_CHANNELS)
#         model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])
        
#         # Print model summary
#         if Summary_param:
#             model.summary()
            
#         # Load the trained weights to the defined model 
#         model.load_weights(model_folder + Save_name + '.hdf5')

#         # Load model history
#         file_name = history_folder + 'History_' + Save_name + '.json'
#         try:
#             with open(file_name, 'r') as f:
#                 content = f.read()
#                 if not content:
#                     print("Error: The file is empty.")
#                 else:
#                     try:
#                         history = json.loads(content)
#                         print("History loaded successfully")
#                     except json.JSONDecodeError as e:
#                         print("JSONDecodeError:", e)
#         except FileNotFoundError:
#             print(f"Error: The file {file_name} does not exist.")
#         except Exception as e:
#             print("An unexpected error occurred:", e)

#         # Create a Simple Wrapper Class:
#         class HistoryWrapper:
#             def __init__(self, history_dict):
#                 self.history = history_dict

#         history = HistoryWrapper(history)

#     else:  
#         # Training model 
#         # Define model
#         model = multi_attention_unet_model(n_classes    = n_classes, 
#                                  IMG_HEIGHT   = IMG_HEIGHT, 
#                                  IMG_WIDTH    = IMG_WIDTH, 
#                                  IMG_CHANNELS = IMG_CHANNELS)
#         model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])
        
#         # Print model summary
#         if Summary_param:
#             model.summary()
        
#         # Train the model
#         history = model.fit(X_train, y_train_cat, 
#                             batch_size = batch_size, 
#                             verbose = 1, 
#                             epochs = epochs, 
#                             validation_data = (X_val, y_val_cat)) #,
#                             # sample_weight = sample_weights)   

#         # Save model and history
#         model.save(model_folder + Save_name + '.hdf5')
#         print(f"Model successfully saved to {model_folder + Save_name + '.hdf5'}")

#         file_path = history_folder + 'History_' + Save_name + '.json'
#         with open(file_path, 'w') as f:
#             json.dump(history.history, f)
#             print(f"History data successfully saved to {file_path}")

#     return model, history

# "=============================================================================================================================================================================================="

