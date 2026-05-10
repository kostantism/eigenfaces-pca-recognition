import numpy as np
import matplotlib.pyplot as plt


IMG_HEIGHT = 112
IMG_WIDTH = 92

p_values = [2, 5, 20, 30, 50]


X_train = np.load("train_centered.npy")   

print("Training data shape:", X_train.shape)


U, S, Vt = np.linalg.svd(X_train, full_matrices=False)

print("U shape:", U.shape)
print("S shape:", S.shape)
print("Vt shape:", Vt.shape)


np.save("U.npy", U)
np.save("S.npy", S)
np.save("Vt.npy", Vt)

print("SVD matrices saved.")


num_to_show = 4  

plt.figure(figsize=(8, 4))
for i in range(num_to_show):
    eigenface = U[:, i].reshape((IMG_HEIGHT, IMG_WIDTH))
    plt.subplot(1, num_to_show, i + 1)
    plt.imshow(eigenface, cmap='gray')
    plt.title("Eigenface {}".format(i + 1))
    plt.axis('off')

plt.suptitle("First Eigenfaces")



for p in p_values:
    Wp = U[:, :p]  
    np.save("eigenfaces_p{}.npy".format(p), Wp)
    print("Saved eigenfaces for p =", p)

plt.show()