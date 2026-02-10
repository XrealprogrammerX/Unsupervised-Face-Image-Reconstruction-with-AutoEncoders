# 🧠 AutoEncoder using PyTorch (LFWPeople Dataset)

This project demonstrates how to build and train a **simple AutoEncoder** using the **PyTorch** deep learning library.  
The AutoEncoder is trained on face images from the **LFWPeople** dataset provided by **Scikit-learn**.

The objective is to learn a **compressed latent representation** of face images and reconstruct them with minimal loss.

---

## 📌 Project Overview

- Implemented a basic AutoEncoder architecture
- Uses PyTorch for model definition and training
- Uses the inbuilt **LFWPeople** dataset from `sklearn`
- Performs image reconstruction
- Suitable for beginners learning representation learning

---
## 🧩 Project Structure

The repository has two main files:

### 1️⃣ Simple Autoencoder
- Uses fully-connected (**Linear**) layers  
- Images are **flattened** before being passed to the network  
- Helps understand:
  - Encoder–decoder structure
  - Latent space compression
  - Reconstruction loss (**MSE**)

### 2️⃣ Convolutional Autoencoder
- Uses **Convolutional** and **Transposed Convolutional** layers  
- Preserves **spatial information** in images  
- Produces **sharper and more realistic reconstructions**  
- Demonstrates why **CNN-based Autoencoders** are preferred for image data


---
## 📂 Dataset

**LFWPeople (Labeled Faces in the Wild)**  
Fetched directly using Scikit-learn:

```python
from sklearn.datasets import fetch_lfw_people
