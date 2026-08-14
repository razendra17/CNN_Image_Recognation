# 🐱🐶 Cats vs Dogs — CNN Image Classification

A simple **Deep Learning image classification project** using **TensorFlow/Keras** and **Convolutional Neural Network (CNN)** to classify images into two categories:

* 🐱 **Cat**
* 🐶 **Dog**

This project was created as part of my journey learning **Artificial Intelligence, Machine Learning, and Computer Vision with Python**.

---

## 📌 Project Overview

The model learns visual patterns from cat and dog images and uses those patterns to classify a new image.

The overall workflow is:

```text
Dataset
   │
   ▼
Image Loading
   │
   ▼
Image Rescaling
   │
   ▼
Convolutional Layer
   │
   ▼
Max Pooling
   │
   ▼
Convolutional Layer
   │
   ▼
Max Pooling
   │
   ▼
Flatten
   │
   ▼
Dense Layer
   │
   ▼
Sigmoid Output
   │
   ▼
🐱 Cat / 🐶 Dog
```

---

## 🧠 Technologies

| Technology | Purpose                             |
| ---------- | ----------------------------------- |
| 🐍 Python  | Programming language                |
| TensorFlow | Deep Learning framework             |
| Keras      | Neural network API                  |
| OpenCV     | Image processing                    |
| NumPy      | Numerical computation               |
| Matplotlib | Training visualization              |
| CUDA / GPU | Accelerated training (if available) |

---

## 📂 Project Structure

```text
cats-vs-dogs/
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   │
│   └── validation/
│       ├── cats/
│       └── dogs/
│
├── training.py
├── predict.py
├── gputest.py
├── cats_vs_dogs.keras
├── dog6.jpg
└── README.md
```

> The `dataset/` folder is not included in this repository if the dataset is too large.

---

# 🏗️ CNN Architecture

The model uses a simple Convolutional Neural Network.

### 1. Rescaling

Images originally have pixel values between:

```text
0 - 255
```

They are normalized into:

```text
0 - 1
```

using:

```python
Rescaling(1./255)
```

---

### 2. Convolutional Layer

The first convolutional layer uses:

```text
Filters      : 32
Kernel       : 3 × 3
Activation   : ReLU
Padding      : Same
```

This layer learns basic visual features such as:

* edges
* shapes
* textures
* simple patterns

---

### 3. Max Pooling

A `2 × 2` MaxPooling layer reduces the spatial dimensions of the image while keeping important features.

```text
224 × 224
      ↓
112 × 112
```

---

### 4. Second Convolutional Layer

The second convolutional layer uses:

```text
Filters      : 64
Kernel       : 3 × 3
Activation   : ReLU
```

With more filters, the network can learn more complex patterns.

---

### 5. Flatten

The extracted feature maps are converted into a one-dimensional vector.

```text
Feature Maps
     ↓
Flatten
     ↓
1D Vector
```

---

### 6. Dense Layer

The model then uses:

```text
128 neurons
Activation: ReLU
```

This layer combines the visual features learned by the CNN.

---

### 7. Output Layer

The final layer contains one neuron:

```python
Dense(1, activation="sigmoid")
```

Because this is a binary classification problem, the sigmoid output represents the probability of one class.

```text
Probability > 0.5 → Dog 🐶
Probability ≤ 0.5 → Cat 🐱
```

---

# ⚙️ Training Configuration

The model is compiled using:

```text
Optimizer : Adam
Loss      : Binary Crossentropy
Metric    : Accuracy
Epochs    : 10
Batch Size: 32
Image Size: 224 × 224
```

Training and validation data are loaded using:

```python
image_dataset_from_directory()
```

This means the folder structure itself determines the class labels.

For example:

```text
dataset/train/
├── cats/
└── dogs/
```

---

# 📊 Training Visualization

The training script plots:

* Training Accuracy
* Validation Accuracy

This makes it easier to observe whether the model is learning properly or potentially **overfitting**.

Example:

```python
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.show()
```

---

# 🔍 Making Predictions

After training, the model is saved as:

```text
cats_vs_dogs.keras
```

The `predict.py` script loads the trained model and processes a new image.

Example:

```python
model = load_model("cats_vs_dogs.keras")
```

The image is resized to:

```text
224 × 224
```

and passed to the model.

Example output:

```text
Dog 🐶
[[0.9823412]]
```

The value represents the model's prediction score.

---

# 🖥️ GPU Detection

The project also contains `gputest.py` to check whether TensorFlow can detect an available GPU.

```python
import tensorflow as tf

print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')

print("Available GPUs:", gpus)
```

Example:

```text
Built with CUDA: True
Available GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If no GPU is detected, TensorFlow will train using the CPU.

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cats-vs-dogs.git
cd cats-vs-dogs
```

---

## 2. Install dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

## 3. Prepare the dataset

Create the following structure:

```text
dataset/
├── train/
│   ├── cats/
│   └── dogs/
│
└── validation/
    ├── cats/
    └── dogs/
```

Place the corresponding images into each folder.

---

## 4. Check GPU

Run:

```bash
python gputest.py
```

---

## 5. Train the model

Run:

```bash
python training.py
```

After training finishes, the model will be saved as:

```text
cats_vs_dogs.keras
```

---

## 6. Predict an image

Put an image in the project directory and change:

```python
imgSrc = "dog6.jpg"
```

Then run:

```bash
python predict.py
```

Example:

```text
Dog 🐶
[[0.9748215]]
```

---

This architecture is intentionally kept simple because the main goal of the project is to understand the fundamentals of **CNN-based image classification**.

---

⭐ If you find this project useful or interesting, feel free to star the repository!
