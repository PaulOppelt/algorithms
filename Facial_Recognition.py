import numpy as np
import matplotlib.pyplot as plt
import torch


# implement algorithm to find closest match of test image in a database using comparrison on a 
# dimension reduced feature space. dimension is reduced using PCA

def facial_recognition(features, labels, test_image, n_components=64, return_image=False):
    """args:
        features: np.array - shape (n_samples, n_features e.g picture (28*28))
        labels: np.array - shape (n_samples,)
        test_image: np.array - shape (n_features,)
        n_components: int - number of components to keep in the PCA decomposition

    return:
        int: index of the closest image in features
        int: label of the closest image in features
    """
    # flatten the image into a 1D array
    features = features.reshape(features.shape[0], -1)
    test_image = test_image.reshape(-1)
    # center the images
    mean_feature = np.mean(features, axis=0)
    centerd_test_image = test_image - mean_feature
    centered_features = features - mean_feature
    # calculate the covariance matrix of the features. The eigen vectors of the covariance matrix are
    # the directions of largest correlation in the data. These directions are the principal components.
    # equivalent: 1-pc = argmax_v1 sum_i (v1 @ x_i)^2
    cov = centered_features.T @ centered_features
    _,vec = np.linalg.eig(cov)
    top_n = vec[:,0:n_components]
    # procet the images using pc. The new features are the  component wise correlations between the images and the principal
    # compoments and the images. 
    projected_features = features @ top_n
    projected_test_image = centerd_test_image @ top_n
    max = np.argmax(projected_features @ projected_test_image)

    if return_image:
        ax, fig = plt.subplots(1, 2)
        fig[0].imshow(test_image.reshape(28, 28), cmap='gray')
        fig[1].imshow(features[max].reshape(28, 28), cmap='gray')
        fig[0].set_title('Test Image')
        fig[1].set_title('Closest Image')
        plt.show()
    return max, labels[max]

