import numpy as np
import matplotlib.pyplot as plt
import os

p_values = [2, 5, 20, 30, 50, 100, 300, 500, 1000]

train_centered = np.load("train_centered.npy")   
test_centered = np.load("test_centered.npy")     
mean_face = np.load("mean_face.npy")

U = np.load("U.npy")   

errors = []

for p in p_values:
    p_eff = min(p, U.shape[1])  
    W = U[:, :p_eff]            
    test_proj = np.dot(W.T, test_centered)
    test_recon = np.dot(W, test_proj) + mean_face[:, np.newaxis]

    error = np.linalg.norm(test_recon - (test_centered + mean_face[:, np.newaxis]))
    errors.append(error)

    print(f"p = {p} (effective {p_eff}) -> Reconstruction error: {error}")

os.makedirs("reconstruction_plots", exist_ok=True)

plt.figure()
plt.plot(p_values, errors, marker='o')
plt.xlabel("Number of Eigenfaces (p)")
plt.ylabel("Total Reconstruction Error")
plt.title("Reconstruction Error vs p")
plt.grid(True)
plt.savefig("reconstruction_plots/reconstruction_error.png")
plt.close()

os.makedirs("reconstructions", exist_ok=True)

image_shape = (112, 92)

persons = [0, 1, 2, 3, 4]  
p_recon = [2, 20, 50, 100, 500]

for person in persons:
    idx = person * 3   
    original = test_centered[:, idx] + mean_face

    plt.imsave(
        f"reconstructions/person_{person+1}_original.png",
        original.reshape(image_shape),
        cmap="gray"
    )

    for p in p_recon:
        p_eff = min(p, U.shape[1])
        W = U[:, :p_eff]

        proj = np.dot(W.T, test_centered[:, idx])
        recon = np.dot(W, proj) + mean_face

        plt.imsave(
            f"reconstructions/person_{person+1}_recon_p{p}.png",
            recon.reshape(image_shape),
            cmap="gray"
        )
