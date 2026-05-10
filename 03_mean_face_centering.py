import numpy as np
import matplotlib.pyplot as plt


train_data = np.load("train_data.npy")   
test_data = np.load("test_data.npy")     

IMG_HEIGHT = 112
IMG_WIDTH = 92


mean_face = np.mean(train_data, axis=1) 

print("Mean face shape:", mean_face.shape)


train_centered = train_data - mean_face[:, np.newaxis]
test_centered = test_data - mean_face[:, np.newaxis]

print("Centered train data shape:", train_centered.shape)
print("Centered test data shape:", test_centered.shape)


np.save("mean_face.npy", mean_face)
np.save("train_centered.npy", train_centered)
np.save("test_centered.npy", test_centered)

print("Mean face and centered data saved.")


mean_face_img = mean_face.reshape((IMG_HEIGHT, IMG_WIDTH))

plt.figure(figsize=(4, 5))
plt.imshow(mean_face_img, cmap='gray')
plt.title("Mean Face (Training Set)")
plt.axis('off')
# plt.show()


mean_face_img = mean_face.reshape((IMG_HEIGHT, IMG_WIDTH))

plt.figure(figsize=(4, 5))
plt.imshow(mean_face_img, cmap='gray')
plt.title("Mean Face (Training Set)")
plt.axis('off')

plt.savefig("mean_face.png", bbox_inches='tight')
plt.close()

print("Mean face image saved as mean_face.png")
