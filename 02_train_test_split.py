import numpy as np


t = 0.7  

NUM_CLASSES = 40
IMAGES_PER_CLASS = 10
IMG_SIZE = 10304


database_3d = np.load("database_3d.npy")     
database_2d = np.load("database_2d.npy")     
labels = np.load("labels.npy")                


num_train = int(t * IMAGES_PER_CLASS)
num_test = IMAGES_PER_CLASS - num_train

print("Training images per class:", num_train)
print("Testing images per class:", num_test)


train_data = np.zeros((IMG_SIZE, num_train * NUM_CLASSES))
test_data = np.zeros((IMG_SIZE, num_test * NUM_CLASSES))

train_labels = np.zeros(num_train * NUM_CLASSES, dtype=np.int32)
test_labels = np.zeros(num_test * NUM_CLASSES, dtype=np.int32)


train_counter = 0
test_counter = 0

for class_idx in range(NUM_CLASSES):
    start_idx = class_idx * IMAGES_PER_CLASS
    end_idx = start_idx + IMAGES_PER_CLASS

    class_images = database_2d[:, start_idx:end_idx]

    train_data[:, train_counter:train_counter + num_train] = class_images[:, :num_train]
    train_labels[train_counter:train_counter + num_train] = class_idx + 1
    train_counter += num_train

    test_data[:, test_counter:test_counter + num_test] = class_images[:, num_train:]
    test_labels[test_counter:test_counter + num_test] = class_idx + 1
    test_counter += num_test


print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("Train labels shape:", train_labels.shape)
print("Test labels shape:", test_labels.shape)


np.save("train_data.npy", train_data)
np.save("test_data.npy", test_data)
np.save("train_labels.npy", train_labels)
np.save("test_labels.npy", test_labels)

print("Train/Test split completed and saved.")
