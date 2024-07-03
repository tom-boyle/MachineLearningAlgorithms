import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample data
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9],
              [1.9, 2.2], [3.1, 3.0], [2.3, 2.7],
              [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

# Create and fit the model
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

# Plot the principal components
plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()