import os
import numpy as np


DATASET_PATH = "orl_faces"

NUM_CLASSES = 40
IMAGES_PER_CLASS = 10

IMG_HEIGHT = 112
IMG_WIDTH = 92
IMG_SIZE = IMG_HEIGHT * IMG_WIDTH  


def read_pgm(filename):
    
    with open(filename, 'rb') as f:
        magic_number = f.readline().strip()
        if magic_number != b'P5':
            raise ValueError("Not a valid PGM P5 file: {}".format(filename))

        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        width, height = map(int, line.split())

        maxval = int(f.readline())

        image = np.fromfile(f, dtype=np.uint8, count=width * height)
        image = image.reshape((height, width))

    return image



database_4d = np.zeros((IMG_HEIGHT, IMG_WIDTH,
                         IMAGES_PER_CLASS, NUM_CLASSES),
                        dtype=np.uint8)

database_3d = np.zeros((IMG_SIZE,
                         IMAGES_PER_CLASS, NUM_CLASSES),
                        dtype=np.float32)

database_2d = np.zeros((IMG_SIZE,
                         IMAGES_PER_CLASS * NUM_CLASSES),
                        dtype=np.float32)

labels = np.zeros(IMAGES_PER_CLASS * NUM_CLASSES, dtype=np.int32)


img_counter = 0

for class_idx in range(NUM_CLASSES):
    person_folder = os.path.join(DATASET_PATH, "s{}".format(class_idx + 1))

    for img_idx in range(IMAGES_PER_CLASS):
        img_path = os.path.join(person_folder, "{}.pgm".format(img_idx + 1))

        if not os.path.exists(img_path):
            raise IOError("File not found: {}".format(img_path))

        img = read_pgm(img_path)

        database_4d[:, :, img_idx, class_idx] = img

        img_vector = img.reshape(IMG_SIZE, 1)

        database_3d[:, img_idx, class_idx] = img_vector[:, 0]

        database_2d[:, img_counter] = img_vector[:, 0]

        labels[img_counter] = class_idx + 1

        img_counter += 1


print("Database construction completed.")
print("database_4d shape:", database_4d.shape)
print("database_3d shape:", database_3d.shape)
print("database_2d shape:", database_2d.shape)
print("labels shape:", labels.shape)


np.save("database_4d.npy", database_4d)
np.save("database_3d.npy", database_3d)
np.save("database_2d.npy", database_2d)
np.save("labels.npy", labels)

print("Databases saved successfully.")
