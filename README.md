# Face Recognition using Eigenfaces (PCA)

A comprehensive face recognition system implemented in Python, based on the **Eigenfaces** method. The project utilizes **Principal Component Analysis (PCA)** for dimensionality reduction and **Nearest Neighbor (NN)** classification for person identification using the ORL Face Database.

## Features

* **Database Building:** Automated conversion and vectorization of PGM facial images.
* **PCA Implementation:** Singular Value Decomposition (SVD) to extract primary facial features (Eigenfaces).
* **Face Recognition:** Support for both Euclidean and Cosine distance metrics.
* **Face Reconstruction:** Ability to reconstruct original faces using a limited number of principal components.
* **Performance Analysis:** Evaluation of accuracy based on different numbers of eigenfaces ($p$) and threshold values ($t$).

## Tech Stack

* **Language:** Python 
* **Libraries:** `NumPy` (Linear Algebra), `Matplotlib` (Visualization)
* **Dataset:** ORL Face Database (40 subjects, 10 images each)

## Project Structure & Execution

The project is modularized into sequential scripts. To run the full pipeline, execute the scripts in the following order:

1.  **`01_build_database.py`**: Reads the ORL dataset and saves it as NumPy arrays.
2.  **`02_train_test_split.py`**: Splits the data into training (70%) and testing (30%) sets.
3.  **`03_mean_face_centering.py`**: Calculates the "mean face" and performs data centering.
4.  **`04_pca_eigenfaces.py`**: Computes SVD to find the U, S, and V matrices and saves the top $p$ eigenfaces.
5.  **`04b_visualize_eigenfaces.py`**: Generates visual representations of the first eigenfaces (ghost faces).
6.  **`05_projection_and_recognition.py`**: Projects test images onto the face space and performs recognition.
7.  **`05b_distance_heatmap.py`**: Visualizes the distance matrix between test and training images.
8.  **`06_face_reconstruction.py`**: Reconstructs faces using $p$ components and calculates error metrics.

## Key Results

### Recognition Accuracy
The system achieves peak performance as the number of eigenfaces ($p$) increases. 
* **Euclidean Distance:** Reliable for standard recognition.
* **Cosine Distance:** Often provides better results in varying lighting conditions.

### Reconstruction Error
The total reconstruction error decreases exponentially as more eigenfaces are used, reaching a plateau when the information gain becomes marginal.

## Installation

1.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib
    ```

2.  **Run the analysis:**
    ```bash
    python 01_build_database.py
    python 02_train_test_split.py
    python 03_mean_face_centering.py
    python 04_pca_eigenfaces.py
    python 05_projection_and_recognition.py
    python 05b_distance_heatmap.py
    python 06_face_reconstruction.py
    ```