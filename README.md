# **Benchmark-for-Semantic-Bridge-Damage-Segmentation**


This project implements semantic segmentation for bridge damage detection using the dacl10k dataset. The model leverages **Feature Pyramid Network (FPN)** and **U-Net** architectures with **EfficientNet** and **ResNet** backbones, supported by the 'segmentation_models' library.

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
You can find the original dataset through this link. https://datasetninja.com/dacl10k#download

---

## **Model Architecture**

The model architecture is based on a Feature Pyramid Network (FPN) with an EfficientNet-B4 backbone, designed for semantic segmentation. The key components of the architecture are:

- **EfficientNet-B4 Backbone** to extract multi-scale feature representations.
- **FPN Layers** to aggregate features across different scales, enabling the model to capture both high- and low-level information.
- **Upsampling Layers** to gradually restore spatial resolution for precise pixel-wise classification.
- The final layer uses **softmax** activation to classify each pixel into one of 19 damage categories.
  
The model uses the following architecture:
```plaintext
Input (512x512x3)
↓
EfficientNet-B4 Backbone (Pretrained on ImageNet)
↓
FPN Layers (Feature Pyramid with multiple scales)
↓
Upsampling Path → Conv2D Blocks for Feature Restoration
↓
Conv2D(19) → Softmax (Pixel-wise Classification into 19 damage categories)
```
This architecture provides efficient multi-scale feature extraction and upsampling for high-resolution segmentation, making it suitable for bridge damage detection tasks in the dacl10k dataset.

---

## **Installation**

To run the project, first clone this repository and install the required dependencies.

### **Clone the repository:**
```bash
git clone https://github.com/AvindaShamal/semantic-bridge-damage-segmentation.git
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

## **Training the Model**
#### Prerequisites
1. Set up the environment:
```bash
%env SM_FRAMEWORK=tf.keras
%pip install segmentation_models
```
2. Load required libraries:
```bash
import tensorflow as tf
from segmentation_models import Unet, FPN
```
#### Model Configuration
The project uses FPN and Unet architectures with EfficientNet-B2, EfficientNet-B4, and ResNet101 backbones. The architecture can be selected by specifying the architecture and backbone parameters in the code.
```bash
model, preprocess_input = create_model("FPN", "efficientnetb4", input_shape=(256, 256, 3), num_classes=19)
```
#### Loss Function
The model uses a 'Combined Loss' function, which is a mix of 'Dice Loss' (to account for class imbalance) and 'Categorical Cross-Entropy' Loss for auxiliary training:

Data Generator
A custom 'Data_generator' function loads images and corresponding mask files in batches for training. This generator preprocesses images and masks from specified directories, ensuring they match in number and format.

#### Training Script
To start model training, use the following command:
```bash
python model.ipynb
```
This command will:

- Load preprocessed data using Data_generator() from the specified directories.
- Train the model using Adam optimizer with a combined Dice and Cross-Entropy loss for auxiliary training.

Training Parameters
- Epochs: Default is set to 20 but can be modified.
- Batch Size: Defined as 16 for this model.
- Optimizer: Adam optimizer with a learning rate of 1e-4.
  
The model will load the preprocessed training data using a generator that fetches images and corresponding masks from disk. It will then train the FPN model with EfficientNetB4 backbone for 20 epochs by default. You can modify the number of epochs, batch size, and other parameters in the script.

```bash
history = model.fit(train_generator, steps_per_epoch=len(train_images) // batch_size, epochs=20, callbacks=[early_stopping, model_checkpoint]
```
The training process displays:

- Epoch progress (e.g., 1/20, 2/20, etc.)
- Accuracy and IoU metrics for each epoch.

---

## **Model Evaluation**

After training, evaluate the model on test data using the 'load_test_batch function' and calculate metrics such as **Mean IoU** and **Mean Accuracy**.

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
# Visualizing results
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

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
