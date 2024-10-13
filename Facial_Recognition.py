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
    features = features.reshape(features.shape[0], -1)
    test_image = test_image.reshape(-1)
    mean_feature = np.mean(features, axis=0)
    centerd_test_image = test_image - mean_feature
    centered_features = features - mean_feature
    cov = centered_features.T @ centered_features
    _,vec = np.linalg.eig(cov)
    top_n = vec[:,0:n_components]
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

if __name__ == "__main__":
    facial_recognition(train['features'], train['labels'], test['features'][60], return_image=True)
