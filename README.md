# 🖼️ Image Classification using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-ANN-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Task-Multi--Class%20Classification-green?style=for-the-badge" />
</p>

<p align="center">
  <b>Handwritten Digit Classification using an Artificial Neural Network</b>
</p>

---

## 📌 Project Overview

This project implements a **Deep Learning based Image Classification system** using an **Artificial Neural Network (ANN)**.

The model is trained to recognize handwritten digits from **0 to 9** using pixel-level image data.

Each image is represented as an **8 × 8 grayscale image**, resulting in **64 pixel features**.

The project demonstrates the complete Deep Learning workflow:

```text
Image Dataset
      ↓
Data Preprocessing
      ↓
Train / Test Split
      ↓
Pixel Normalization
      ↓
ANN Architecture
      ↓
Forward Propagation
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
RMSProp Optimization
      ↓
Model Training
      ↓
Evaluation
      ↓
Prediction
      ↓
Saved Model
```

---

## 🎯 Objective

The main objective of this project is to build a neural network that can classify handwritten digit images into one of **10 classes**:

```text
0  1  2  3  4  5  6  7  8  9
```

---

## 📊 Dataset

The project uses the **Scikit-learn Digits dataset**.

### Dataset Details

| Property       |     Value |
| -------------- | --------: |
| Total Images   |     1,797 |
| Image Size     |     8 × 8 |
| Pixel Features |        64 |
| Classes        |        10 |
| Classes        |       0–9 |
| Image Type     | Grayscale |

The dataset is converted into a CSV format for preprocessing and model training.

### Dataset Structure

```text
pixel_0
pixel_1
pixel_2
...
pixel_63
label
```

The `label` column represents the actual digit.

---

## 🧠 Deep Learning Model

The project uses a fully connected **Artificial Neural Network (ANN)**.

### Architecture

```text
Input Layer
64 Features
     ↓
Dense Layer
128 Neurons
ReLU
     ↓
Dense Layer
64 Neurons
ReLU
     ↓
Output Layer
10 Neurons
Softmax
```

### Model Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation="relu", input_shape=(64,)),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
```

---

## ⚙️ Model Compilation

The model uses **RMSProp** as the optimizer and **Sparse Categorical Crossentropy** as the loss function.

```python
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Why RMSProp?

RMSProp adapts the learning rate for different parameters by considering the recent history of squared gradients.

This makes it useful for neural network training.

---

## 🔄 Data Preprocessing

### 1. Feature and Target Separation

```python
X = df.drop("label", axis=1)
y = df["label"]
```

### 2. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 3. Pixel Normalization

The original pixel values range from **0 to 16**.

Therefore, the values are normalized to approximately **0 to 1**:

```python
X_train = X_train / 16.0
X_test = X_test / 16.0
```

---

## 🚀 Model Training

The ANN is trained using:

* Epochs: `20`
* Batch Size: `32`
* Validation Split: `0.2`

```python
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)
```

During training, the model performs:

```text
Forward Propagation
       ↓
Prediction
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Gradient Calculation
       ↓
RMSProp
       ↓
Weight Update
```

This process is repeated for every batch across multiple epochs.

---

## 📉 Training & Validation Loss

The training history is visualized to understand how the model learns.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
```

---

## 📈 Training & Validation Accuracy

```python
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()
```

These graphs help analyze whether the model is learning effectively and whether there are signs of overfitting.

---

## 🧪 Model Evaluation

The trained model is evaluated using the test dataset.

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

---

## 🔮 Prediction

The model produces 10 probability values using the **Softmax** activation function.

```python
y_pred_prob = model.predict(X_test)

y_pred = np.argmax(y_pred_prob, axis=1)
```

`argmax()` selects the class having the highest probability.

Example:

```text
Softmax Output:

[0.00, 0.01, 0.00, 0.02, 0.01,
 0.93, 0.01, 0.00, 0.01, 0.01]

Highest Probability → Class 5

Prediction → 5
```

---

## 🖼️ Visual Prediction

The 64 pixel values can be reshaped back into an **8 × 8 image**.

```python
image = X_test.iloc[0].values.reshape(8, 8)

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
```

Actual and predicted values can be compared:

```python
print("Actual:", y_test.iloc[0])
print("Predicted:", y_pred[0])
```

---

## 📊 Confusion Matrix

A confusion matrix is used to analyze classification performance across all ten digit classes.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.show()
```

The diagonal represents correct predictions, while off-diagonal values represent misclassifications.

---

## 📋 Classification Report

Detailed classification metrics can be generated using:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

The report includes:

* Precision
* Recall
* F1-score
* Support

---

## 💾 Saving the Model

After training, the model is saved in Keras format:

```python
model.save("digit_classifier.keras")
```

The saved model can later be loaded without retraining:

```python
from tensorflow.keras.models import load_model

loaded_model = load_model("digit_classifier.keras")
```

---

## 🔁 Prediction using the Saved Model

```python
index = 25

sample_image = X_test.iloc[index].values.reshape(1, 64)

probabilities = loaded_model.predict(
    sample_image,
    verbose=0
)

predicted_digit = np.argmax(probabilities)

print("Actual Digit   :", y_test.iloc[index])
print("Predicted Digit:", predicted_digit)
```

---

## 🗂️ Project Structure

```text
Image-Classification-Deep-Learning/
│
├── data/
│   └── digits_image_classification.csv
│
├── notebooks/
│   └── image_classification.ipynb
│
├── digit_classifier.keras
├── digits_image_classification_dataset.zip
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* Jupyter Notebook
* VS Code

---

## 🧠 Concepts Demonstrated

This project provides practical implementation of:

* Artificial Neural Networks
* Dense Layers
* Neurons
* Weights
* Bias
* Forward Propagation
* Activation Functions
* ReLU
* Softmax
* Loss Function
* Sparse Categorical Crossentropy
* Backpropagation
* Gradients
* RMSProp
* Epochs
* Batches
* Validation Data
* Model Evaluation
* Classification
* Confusion Matrix
* Precision
* Recall
* F1-score
* Model Saving and Loading

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Image-Classification-Deep-Learning
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
notebooks/image_classification.ipynb
```

Run the cells from top to bottom.

---

## 📌 Future Improvements

The current project uses a fully connected ANN.

Possible future improvements include:

* CNN-based image classification
* Data augmentation
* Hyperparameter tuning
* Dropout
* Batch Normalization
* Learning-rate scheduling
* Transfer Learning
* Real handwritten image input
* Web/API deployment
* Streamlit interface
* TensorFlow Lite deployment

---

## 👨‍💻 Author

**Shaik Yusufvali**

AI & Machine Learning Engineer
Deep Learning | Generative AI | Robotics AI

---

## ⭐ Project Goal

This project is part of a practical **Deep Learning learning journey**, progressing from neural network fundamentals toward advanced architectures such as:

```text
ANN
 ↓
CNN
 ↓
RNN
 ↓
LSTM / GRU
 ↓
Attention
 ↓
Transformers
 ↓
Generative AI
```

---

## 📜 License

This project is intended for educational and portfolio purposes.
