# Plant-Diseases-Classification-with-CNN-7-classes-91-percentage-test-accuracy

## Introduction
Plant disease detection plays a crucial role in agriculture, as early identification enables timely preventive measures, minimizes crop losses, and promotes better productivity. In this project, we aimed to develop a deep learning model based on Convolutional Neural Networks (CNN) to classify plant diseases. By utilizing a multiclass classification approach, this model can assist agricultural professionals in detecting diseases early, facilitating rapid interventions to safeguard crops.

## Dataset
The dataset comprises 3,224 images, categorized into 7 plant species with their corresponding diseases. The plant species and diseases are as follows:

| Plant                | Disease                                  |
|----------------------|------------------------------------------|
| Apple                | Cedar apple rust                         |
| Corn (maize)         | Cercospora leaf spot, Gray leaf spot     |
| Grape                | Black rot                                |
| Orange               | Haunglongbing (Citrus greening)          |
| Potato               | Early blight                             |
| Strawberry           | Leaf scorch                              |
| Tomato               | Leaf Mold                                |

## Data Preprocessing and Augmentation
To improve the generalization ability and mitigate overfitting, the following data augmentation techniques were applied:
- **Width and Height Shift**: 30%
- **Shear and Zoom**: 30%
- **Horizontal and Vertical Flipping**: Applied to increase image variability.
- **Rescaling**: Pixel values were normalized to the [0,1] range for improved training efficiency.

These techniques helped enhance the model's ability to generalize well to unseen data.

## Model Architecture
The model was designed using a CNN architecture, incorporating L2 regularization to reduce overfitting and improve generalization. The architecture consists of the following layers:
- **Input Image Size**: 128x128 pixels
- **Convolutional Layers**: ReLU activation functions for feature extraction
- **MaxPooling Layers**: To downsample and reduce spatial dimensions
- **Dropout Layer**: 50% dropout to reduce overfitting by randomly setting input units to 0
- **Dense Layers**: For classification with Softmax activation for multiclass classification
- **L2 Regularization**: Applied on Dense layers to penalize large weights and mitigate overfitting

## Hyperparameters
The model was trained with the following hyperparameters:
- **Optimizer**: Adam optimizer with a learning rate of 0.001
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 10
- **Dropout Rate**: 0.5
- **L2 Regularization**: A regularization penalty coefficient of 0.01

## Model Training
The model was trained using the Adam optimizer and categorical cross-entropy loss. Training details are as follows:
- **Epochs**: 50 with early stopping and learning rate adjustments
- **Batch Size**: 32
- **Regularization**: Dropout and L2 penalties were applied

## Evaluation Metrics
Model performance was evaluated using various metrics:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Balances precision and recall
- **Precision**: Measures how many of the predicted positive cases were correct
- **Recall**: Measures how many actual positive cases were identified
- **Confusion Matrix**: Class-wise performance analysis

## Classification Report
| **Class**                                | **Precision** | **Recall** | **F1-Score** | **Support** |
|------------------------------------------|---------------|------------|--------------|-------------|
| Apple - Cedar apple rust                 | 0.98          | 0.85       | 0.91         | 440         |
| Corn (maize) - Cercospora leaf spot      | 0.93          | 0.80       | 0.86         | 410         |
| Grape - Black rot                        | 0.89          | 0.93       | 0.91         | 472         |
| Orange - Haunglongbing (Citrus greening) | 0.99          | 0.89       | 0.94         | 503         |
| Potato - Early blight                    | 0.78          | 1.00       | 0.88         | 485         |
| Strawberry - Leaf scorch                 | 0.92          | 0.98       | 0.95         | 444         |
| Tomato - Leaf Mold                       | 0.96          | 0.93       | 0.94         | 470         |
| **Accuracy**                             |               |            | 0.91         | 3224        |
| **Macro Avg**                            | 0.92          | 0.91       | 0.91         | 3224        |
| **Weighted Avg**                         | 0.92          | 0.91       | 0.91         | 3224        |

## Results
The model achieved a **validation accuracy of 91%**, and the F1-scores for individual classes are as follows:
- Apple - Cedar apple rust: F1-Score = 0.91
- Corn (maize) - Cercospora leaf spot, Gray leaf spot: F1-Score = 0.86
- Grape - Black rot: F1-Score = 0.91
- Orange - Haunglongbing (Citrus greening): F1-Score = 0.94
- Potato - Early blight: F1-Score = 0.88
- Strawberry - Leaf scorch: F1-Score = 0.95
- Tomato - Leaf Mold: F1-Score = 0.94

The **top three diseases detected accurately** were:
- Strawberry - Leaf scorch
- Orange - Haunglongbing (Citrus greening)
- Tomato - Leaf Mold

### Future Work
- **Dataset Expansion**: Increasing the dataset size and incorporating more plant species with more tuned images and disease variations will improve model performance.
- **Transfer Learning**: Using pretrained models such as VGG16 or ResNet could further enhance accuracy.
- **Additional Disease Classification**: Including more plant diseases will make the model a more versatile tool for the agricultural industry.

## Conclusion
This project successfully demonstrated the potential of CNNs for the classification of plant diseases. Achieving a 91% validation accuracy, the model provides valuable insights into early disease detection and offers a foundation for developing practical applications in agriculture.

