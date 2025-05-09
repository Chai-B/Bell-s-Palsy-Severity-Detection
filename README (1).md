
# Bell’s Palsy: Automated Detection using Deep Learning

Bell’s Palsy: Automated Detection using Deep Learning

Table of Contents
-


[1. Objectives](#Objectives)

[2. Dataset](#Dataset)

[3. Methodology](#Methodology)

[4. Model Architecture](#ModelArchitecture)

[5. Results](#Results)

[6. Technologies](#Technologies)








## Objectives


The primary goal of this project is to develop a deep learning-based system capable of **automatically detecting and classifying the severity of Bell’s Palsy** using facial image analysis. The specific objectives include:

- **Automated Diagnosis**  
  Design a system that assists healthcare professionals in identifying Bell’s Palsy from facial images, reducing subjectivity and diagnostic time.

- **Severity Classification**  
  Build a model that can categorize facial paralysis into four clinically recognized severity levels:
  - Mild
  - Moderate
  - Moderate Severe
  - Severe

- **Region-Based Analysis**  
  Analyze specific facial regions such as:
  - **Mouth** – to detect drooping or asymmetry during smiling.
  - **Eyes** – to identify incomplete eye closure.
  - **Eyebrows** – to observe elevation disparities.

- **Efficient and Scalable**  
  Create a lightweight, accurate model that can be used in remote or resource-limited environments for early screening.

- **Improve Treatment Planning**  
  Provide detailed severity output to aid in tailoring treatment strategies and tracking patient recovery over time.

## Dataset


- **Source**: YouTube Facial Paralysis (YFP) dataset with 2,019 labeled images across four severity classes: mild, moderate, moderate severe, and severe.

- **Preprocessing**:
  - Images resized to 224×224 pixels
  - Normalized pixel values (0–1 range)
  - Data augmentation: rotation, zoom, shear, shift, brightness/contrast adjustment, horizontal flip

- **Split**:
  - 80% Training
  - 10% Validation
  - 10% Testing



## Methodology

1. **Data Preparation**  
   - Collected and labeled facial images by severity level  
   - Applied preprocessing: resizing, normalization, and augmentation  
   - Dataset split into training, validation, and test sets (80/10/10)

2. **Model Development**  
   - Used transfer learning with **ResNet-50** pretrained on ImageNet  
   - Added a custom classification head for 4 severity classes  
   - Trained using Adam optimizer and categorical cross-entropy loss

3. **Training Strategy**  
   - Batch size: 32, Epochs: 30  
   - Callbacks: EarlyStopping and ReduceLROnPlateau  
   - Fine-tuned last 20 layers of ResNet-50 for improved accuracy

4. **Evaluation**  
   - Measured accuracy, F1-score, and AUC on mouth, eyes, and eyebrow regions  
   

## Model Architecture


- **Base Model**:  
  - **ResNet-50** pretrained on ImageNet  
  - Base layers frozen during initial training for feature reuse

- **Custom Head**:  
  - Global Average Pooling  
  - Dense (256) → Dropout (0.5)  
  - Dense (128) → Dropout (0.3)  
  - Final Dense (4) with Softmax activation for 4-class classification

- **Training Details**:  
  - Optimizer: Adam (initial learning rate = 1e-4)  
  - Loss Function: Categorical Crossentropy  
  - Regularization: Dropout, L2 weight decay, Batch Normalization  
  - Fine-tuning: Last 20 layers of ResNet-50 unfrozen after initial training

## Results



The model was evaluated on separate facial regions (mouth, eyes, eyebrows) to classify Bell’s Palsy severity.

| Region     | Accuracy | AUC   | F1-Score |
|------------|----------|-------|----------|
| Mouth      | 98.79%   | 1.00  | 0.97     |
| Eyes       | 97.50%   | 0.99  | 0.96     |
| Eyebrows   | 94.18%   | 0.96  | 0.94     |

- **Mean F1-Score**: 0.97  
- **Inference Time**: <200ms per image on Tesla T4 GPU

> Data augmentation, transfer learning, and regularization contributed to high model generalization and performance.

## Technologies

***
A list of technologies used within the project:

* [TensorFlow](https://www.tensorflow.org/): Version 2.14.0  
* [Keras](https://keras.io/): Version 2.14.0  
* [OpenCV](https://opencv.org/): Version 4.8.0  
* [Python](https://www.python.org/): Version 3.9  
* [Google Colab](https://colab.research.google.com/): Tesla T4 GPU runtime  
* [ResNet-50 (ImageNet)](https://keras.io/api/applications/resnet/): Pre-trained model  
* [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam): Built-in in Keras  
* [EarlyStopping Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping): Built-in in Keras  
* [ReduceLROnPlateau Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau): Built-in in Keras  
* [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator): Built-in in Keras

