EGG White, Yolk and Aircell detection

This Project tried to estimate the volume of white, yolk and air cell portion of an Egg.

Files:
    - mri_single_egg.nii: NII file of MRI of an egg
    - final_kmean_cluster_model: A trained KMeans clustering model
        - trained with 5 clusters
        - Cluster 3 satisfactorily recognises images with clearly visible \
            egg's outer boundary and inner yolk boundary
    - task2_post_interview.py: script with main function
    - KNN_train.py: helper package for tarining and saving the KMeans model

Calculation of Volume of Yolk:
    - Volume of egg(given) = 60 ml
    - Volume of Yolk = (Pixels in image that represent yolk/ Total no. of pixels) * Volume of egg
        - Pixels in image that represent yolk: Segmented using thresholding
        - Total no. of pixels: Total number of pixels in the image
    - Error in estimation: 'Total no. of pixels' includes pixels from surrounding area of egg

Calculation of Volume of White portion:
    - Volume of egg(given) = 60 ml
    - Volume of Yolk = (Pixels in image that represent white portion/ Total no. of pixels) * Volume of egg
        - Pixels in image that represent white portion: Segmented using thresholding
        - Total no. of pixels: Total number of pixels in the image
    - Error in estimation: 'Total no. of pixels' includes pixels from surrounding area of egg

Calculation of Aircell:
    - TODO

How to use:
 - Unzip folder
 - Install requirements
 - Run 'python task2_post_interview.py'
 - Results:
    - A plot will show the results of KMean clustering model on 20 test images
    - Volume of Egg-white and Egg yolk will be printed in terminal

