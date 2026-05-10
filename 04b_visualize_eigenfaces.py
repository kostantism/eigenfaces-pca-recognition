import numpy as np
import matplotlib.pyplot as plt
import os

IMG_HEIGHT = 112
IMG_WIDTH = 92

U = np.load("U.npy")

num_to_save = 4



output_dir = "eigenfaces_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(num_to_save):
    eigenface = U[:, i].reshape((IMG_HEIGHT, IMG_WIDTH))

    plt.figure(figsize=(3, 4))
    plt.imshow(eigenface, cmap='gray')
    plt.title("Eigenface {}".format(i + 1))
    plt.axis('off')

    filename = os.path.join(output_dir, "eigenface_{}.png".format(i + 1))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

print("Eigenface images saved in folder:", output_dir)
