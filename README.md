# **Benchmark-for-Semantic-Bridge-Damage-Segmentation**


This repository contains the code and resources for performing **Semantic Bridge Damage Segmentation** using the **dacl10k Dataset**. The project focuses on identifying different types of damages on bridges from images and classifying them into multiple categories using a **Convolutional Neural Network (CNN)** model.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## **Project Overview**

The **Semantic Bridge Damage Segmentation** project is a deep learning-based approach to automatically segment and identify different damage categories on bridge images. The model uses **semantic segmentation** techniques to assign a damage category label to each pixel in the image.

The project is built using:
- **TensorFlow** and **Keras** for deep learning.
- The **dacl10k Dataset**, which contains 9,920 annotated images of damaged bridges.
- **Preprocessing** tools to resize, filter, and normalize the images and convert JSON annotations into mask formats for training.

## **Dataset**

### dacl10k Dataset

The **dacl10k Dataset** contains:
- **Images** of damaged bridges.
- **Annotations** in JSON format, which are converted into **masks** that the model can use for segmentation tasks.

The dataset consists of 12 damage classes and 6 bridge components, including cracks, corrosion, and surface wear, each of which is represented by a different class in the segmentation mask.

---

## **Model Architecture**

The model architecture is based on a **fully convolutional network** for semantic segmentation. The key components of the architecture are:
- **Convolutional Layers** to extract image features.
- **MaxPooling Layers** to downsample feature maps.
- **UpSampling Layers** to restore spatial resolution for pixel-wise classification.
- The final layer uses **softmax activation** to classify each pixel into one of 19 damage categories.

The model uses the following architecture:
```plaintext
Input (512x512x3)
↓
Conv2D(32) → ReLU → MaxPooling2D
↓
Conv2D(64) → ReLU → MaxPooling2D
↓
Conv2D(128) → ReLU
↓
UpSampling2D → Conv2D(64) → ReLU
↓
UpSampling2D → Conv2D(32) → ReLU
↓
Conv2D(19) → Softmax (Pixel-wise Classification)
```

---

## **Installation**

To run the project, first clone this repository and install the required dependencies.

### **Clone the repository:**
```bash
git clone https://github.com/yourusername/semantic-bridge-damage-segmentation.git
cd semantic-bridge-damage-segmentation
```

### **Install the required Python packages:**
You can install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

The dependencies include:
- TensorFlow
- Keras
- Numpy
- OpenCV
- Matplotlib

---

## **Usage**

### **Data Preprocessing**

Before training the model, the **dacl10k** dataset needs to be preprocessed. This includes converting the **JSON** annotations to **mask** format (NumPy arrays) and filtering the images for training. You can use the following preprocessing steps:

1. **Annotation Conversion**: Convert the dacl10k dataset annotations from JSON format to mask format using the `labelme2mask` function.
2. **Normalization and Resizing**: Ensure the images are resized to 512x512 pixels and normalized for optimal training performance.

```python
# Example of converting annotations
labelme2mask(json_annotation_path, output_mask_folder)
```

### **Training the Model**

To train the model, run the following command:
```bash
python model.ipynb
```

The model will load the preprocessed training data using a generator that fetches images and corresponding masks from disk. It will then train the CNN model for 15 epochs by default. You can modify the number of epochs, batch size, and other parameters in the script.

---

## **Training the Model**

You can train the model by running the `main` function in `train.py`:

1. The **train.py** script uses the `preprocessed_data_generator()` function to load preprocessed images and masks from the dataset.
2. The model will be trained using the **categorical cross-entropy** loss function and **Adam optimizer**.

### **Expected Output**:
The training process will display the following information:
- Epoch progress (e.g., 1/15, 2/15, etc.)
- Accuracy and loss metrics for each epoch.

---

## **Model Evaluation**

Once training is complete, you can evaluate the model using test data. You can modify the script to load a test set and compute metrics like **IoU** (Intersection over Union) or pixel accuracy.

```python
# Example of evaluating the model on test data
model.evaluate(test_generator, steps=steps_per_epoch)
```

---

## **Results**

After training, the model will produce segmentation maps that assign damage labels to each pixel in the input bridge images. The output masks can be visualized using **Matplotlib** or any image viewer.

Example of visualizing results:
```python
import matplotlib.pyplot as plt

# Visualize an input image and its corresponding mask
plt.imshow(image)
plt.imshow(predicted_mask, alpha=0.5)
plt.show()
```

---

## **Contributing**

We welcome contributions from the community! If you find any issues or want to improve this project, feel free to submit a pull request or open an issue.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

### **Contact**

For any questions or inquiries, feel free to contact me at [avindashamal@gmail.com].
