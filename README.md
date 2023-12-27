# Principal-Component-Analysis-PCA-for-Dimensionality-Reduction
Example using PCA to reduce the dimensionality of a dataset
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)

# Visualize the reduced data
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y, cmap='viridis')
plt.title('PCA for Dimensionality Reduction')
plt.show()
