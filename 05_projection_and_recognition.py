import numpy as np

p_values = [2, 5, 20, 30, 50]
t_values = [0.2, 0.5, 0.7, 0.9]
distance_type = "euclidean"

train_centered = np.load("train_centered.npy")
test_centered = np.load("test_centered.npy")

train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

for p in p_values:
    W = np.load(f"eigenfaces_p{p}.npy")   

    train_proj = np.dot(W.T, train_centered)  
    test_proj = np.dot(W.T, test_centered)    

    print(f"\n=== p = {p} ===")

    for t in t_values:
        correct = 0
        total = test_proj.shape[1]

        for i in range(total):
            test_vec = test_proj[:, i]
            distances = np.linalg.norm(train_proj - test_vec[:, None], axis=0)

            nearest_idx = np.argmin(distances)

            if distances[nearest_idx] <= t * np.mean(distances):
                predicted_label = train_labels[nearest_idx]
            else:
                predicted_label = -1   

            if predicted_label == test_labels[i]:
                correct += 1

        accuracy = correct / total
        print(f"t = {t} -> Recognition rate: {accuracy:.4f}")
