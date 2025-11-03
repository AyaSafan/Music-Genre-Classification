# 🎵 Music Genre Classification using CNNs

This project explores **automatic music genre classification** using **Mel-spectrograms** and a **Convolutional Neural Network (CNN)**.  
It was developed as a deep learning experiment in audio signal processing and music information retrieval.

---

## 📘 Overview

The goal of this project is to train a neural network that can predict the **genre of a music clip** based on its audio characteristics.  
By transforming audio signals into **Mel-spectrograms**, the project leverages the ability of CNNs to learn spatial hierarchies and recognize genre-specific audio patterns.

---

## 🧩 Workflow

1. **Data Loading & Preprocessing**
   - Load audio clips from the dataset.
   - Extract **Mel-spectrograms** using the `librosa` library.
   - Normalize data and prepare it for CNN input.

2. **Model Architecture**
   - A **Convolutional Neural Network (CNN)** designed to process Mel-spectrograms.
   - Layers typically include:
     - 2D Convolution layers with ReLU activations  
     - MaxPooling layers  
     - Dropout for regularization  
     - Fully connected dense layers leading to softmax output

3. **Training & Evaluation**
   - Train the CNN on labeled music samples.
   - Evaluate accuracy, loss, and confusion matrix on the test set.
   - Visualize learning curves for model performance.

4. **Genre Prediction**
   - Once trained, the model can predict the genre of new, unseen audio clips.

---

## 📁 Repository Contents

- `MEL-Music-Genre-Classification.ipynb` — Jupyter Notebook with full implementation  
- `legacy/` — Older experiments 
---

## ⚙️ Requirements

To run the notebook, install the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn librosa tensorflow keras scikit-learn
