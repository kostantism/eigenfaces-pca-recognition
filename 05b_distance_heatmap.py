import numpy as np
import matplotlib.pyplot as plt
import os


p = 20
distance_type = "euclidean"   

train_centered = np.load("train_centered.npy")   
test_centered = np.load("test_centered.npy")   

train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

W = np.load("eigenfaces_p{}.npy".format(p))      


train_proj = np.dot(W.T, train_centered)   
test_proj = np.dot(W.T, test_centered)     


def euclidean_distance_matrix(A, B):
   
    D = np.zeros((A.shape[1], B.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            D[i, j] = np.linalg.norm(A[:, i] - B[:, j])
    return D


def cosine_distance_matrix(A, B):
    D = np.zeros((A.shape[1], B.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            a = A[:, i]
            b = B[:, j]
            D[i, j] = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return D


if distance_type == "euclidean":
    D = euclidean_distance_matrix(test_proj, train_proj)
else:
    D = cosine_distance_matrix(test_proj, train_proj)

print("Distance matrix shape:", D.shape)


output_dir = "heatmaps"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.figure(figsize=(10, 6))
plt.imshow(D, aspect='auto', cmap='hot')
plt.colorbar(label="Distance")
plt.xlabel("Training samples")
plt.ylabel("Testing samples")
plt.title("Distance Matrix Heatmap (p = {}, {})".format(p, distance_type))

filename = os.path.join(
    output_dir,
    "distance_heatmap_p{}_{}.png".format(p, distance_type)
)
plt.savefig(filename, bbox_inches='tight')
plt.close()

print("Heatmap saved as:", filename)
