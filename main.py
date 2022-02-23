"""

Author: Praveen K
Date: Feb 3rd, 2022
Contact: aaa@email.com

This Project tried to estimate the volume of white, yolk and air cell portion of an Egg.
"""

# Library Imports
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pickle
from KNN_train import train_model, save_model # Will be used for training
                                              #  and saving a new KMean model


def load_nifti(input_nifti_file):
    """
    This function loads the nii image
    
    Parameters
    -----------
    input_nifti_file:
        path of the input file
    
    returns
    ------
    nii_data
        the loaded nii data object
    """
    nii_data = nib.load(input_nifti_file)
    return nii_data


def compute_white_volume(useful_images, image_shape):
    """
    This function computes the volume of white portion of egg
    Volume of White portion = (Pixels in image that represent white portion/ Total pixels) * Volume of egg
    Parameters
    ----------
    useful_images: list of images
    image_shape:  Shape of individual images
    
    Returns:
    --------
    white_volume: Volume of White portion of the egg
    """
    # Helper veariables to count pixels
    white_pixel_count = 0
    total_pixel_count = 0
    
    # Iterate through the images to look for Egg White pixels
    for img in useful_images:
        #Reshape images to original size, for viewing during debugging
        img = img.reshape(image_shape[0], image_shape[1])
        # Convert to 0 - 255 range
        img = img * 255
        # Set Yolk pixels as 255(white) and rest as 0(black)
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                if img[i][j]>80:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
        # get count of pixels that are 255
        (_, count) = np.unique(img, return_counts=True)
        # Calculate volume
        if len(count) <= 1:
            pass
        else:
            white_pixel_count += count[1]
            total_pixel_count += count[0] + count[1]
    white_volume = (white_pixel_count/total_pixel_count) * 60
    # return Volume
    return round(white_volume, 3)


def compute_yolk_volume(useful_images, image_shape):
    """
    This function computes the Yolk volume of egg
    Volume of Yolk = (Pixels in image that represent yolk/ Total pixels) * Volume of egg
    Parameters
    ----------
    useful_images: list of images
    image_shape:  Shape of individual images
    
    Returns:
    --------
    yolk_volume: Volume of yolk portion of the egg
    """
    # Helper veariables to count pixels
    yolk_pixel_count = 0
    total_pixel_count = 0

    # Iterate through the images to look for Yolk pixels
    for img in useful_images:
        # Reshape images to original size, for viewing during debugging
        img = img.reshape(image_shape[0], image_shape[1])
        # Convert to 0 - 255 range
        img = img * 255
        # Set Yolk pixels as 255(white) and rest as 0(black)
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                if img[i][j]>30 and img[i][j]<70:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
        # get count of pixels that are 255
        (_, count) = np.unique(img, return_counts=True)
        # Calculate volume
        if len(count) <= 1:
            pass
        else:
            yolk_pixel_count += count[1]
            total_pixel_count += count[0] + count[1]
    yolk_volume = (yolk_pixel_count/total_pixel_count) * 60
    # return Volume
    return round(yolk_volume, 3)

# TODO
def compute_aircell_volume(image):
    # TODO
    pass

def create_dataset(nii_file_data):
    """
    This function creates image dataset from nii file
    Parameters
    ----------
    nii_file_data: loaded nii file 
    Returns:
    --------
    X_train: Train set
    X_test: Test set
    """
    dataset = []
    for i in range((nii_file_data.shape)[2]):
        img = nii_file_data[:,:,i]
        img = np.array(img)
        flatten_img = img.flatten()
        dataset.append(flatten_img)
    dataset = np.array(dataset)
    X_train, X_test = train_test_split(dataset, test_size=0.25, train_size=0.75)
    return X_train, X_test

def plot_predictions(preds, X_test, image_shape):
    """
    This function plots the clustering algorithm's output on test images, and svaes a .png copy
    Parameters
    ----------
    preds: Predicted cluster for test images
    X_test: Test images
    image_shape: shaepe of individual test image
    
    Returns:
    --------
    None
    """
    _, ax = plt.subplots(ncols=5, nrows=4)
    for count, i, ax in zip(range(20), preds, ax.flatten()):
        reshape_image = (X_test[count].reshape(image_shape[0], image_shape[1]))*255
        ax.imshow(reshape_image, cmap='gray', origin='lower')
        ax.axis('off')
        ax.set_title('Cluster = %i' % i, fontsize=10)
    plt.savefig('Cluster_predictions.png')
    plt.show()
    pass

def main():
    # Step 1 - Load data
    img = load_nifti('./mri_single_egg.nii')
    img_data = img.get_fdata()
    # Shape of data
    print("Shape of data: {}".format((img_data.shape)))

    # Create Dataset from nii file and split test and train data
    X_train, X_test = create_dataset(img_data)

    # Name of the Saved model
    filename = './final_kmean_cluster_model'
    # Load Saved Kmeans model
    kmean_cluster = pickle.load(open(filename, 'rb'))

    """
     Uncomment below to train the KNN model,
     NOT Recommended as clusters will change and
     retuning will be required for later part of code
    """
    #kmean_cluster = train_model(X_train) # Uncomment this line
    #save_model(kmean_cluster) # To save a trained model, uncomment this
    
    # Do Predictions on test images
    preds = kmean_cluster.predict(X_test)
    image_shape = [img_data.shape[0], img_data.shape[1]]
    
    # Calculate Volume of White part of Egg
    white_part_images = []
    for count, i in enumerate(preds):
        if i==3:
            white_part_images.append(X_test[count])
    
    white_yolume = compute_white_volume(white_part_images, image_shape)
    print("valume of White Part in egg: {} ml".format(white_yolume))

    # Calculate Volume of yolk of Egg
    yolk_part_images = []
    for count, i in enumerate(preds):
        if i==3:
            yolk_part_images.append(X_test[count])
    yolk_volume = compute_yolk_volume(yolk_part_images, image_shape)
    print("valume of Yolk Part in egg: {} ml".format(yolk_volume))

    # Plot cluster predictions
    plot_predictions(preds, X_test, image_shape)

# Main function
if __name__=='__main__':
    main()



