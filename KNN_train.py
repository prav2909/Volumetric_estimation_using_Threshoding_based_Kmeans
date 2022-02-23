"""
Author: Praveen K
Date: Feb 3rd, 2022
Contact: aaa@email.com

This file trains a KMeans model and saves a Kmeans model
"""

# Library Imports
from sklearn.cluster import KMeans
import pickle



def train_model(dataset):
    """
    Loads a KMeans model

    Parameter
    ---------
    dataset: Dataset of images for training

    Returns
    ---------
    Trained KMeans model with 5 clusters
    """
    kmean_cluster = KMeans(n_clusters=5).fit(dataset)
    return kmean_cluster

def save_model(kmean_cluster_final):
    """
    This function saves a trained KMeans model

    Parameter
    ---------
    kmean_cluster_final: KMean model to save

    Returns:
    --------
    None
    """
    filename = './final_kmean_cluster_model'
    pickle.dump(kmean_cluster_final, open(filename, 'wb'))
    pass
