# COVID-19 Chest X-Ray Image Detection Using Deep Learning

## Project Synopsis
This project leverages deep convolutional neural networks (CNNs) to detect COVID-19 from chest X-ray images, employing a transfer learning approach to refine pre-existing models to this task. 

## Project Foundation
Methodology is inspired by the study ["Deep Learning for Coronavirus (COVID-19) Diagnosis using CT images"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7372265/). By using this research as a bedrock, I aim to replicate the findings with pretrained models.

## Dataset Overview
The COVID-Xray-5k dataset employed in this project comprises a total of 5,184 chest X-ray images, divided into a training set of 2084 images and a test set of 3100 images.

## Accessing the Dataset
For those interested in reviewing the data, the dataset can be accessed via the link provided below: <br>
[COVID-Xray-5k Dataset Download](https://www.dropbox.com/scl/fi/ajy4i9u4bjt4ho3dz4l37/data_upload_v3.zip?rlkey=kyh5oz91vykk7cao6jiip4dyn&e=1&dl=0)

## Data Structure and Organization
```plaintext
data/
├── train/
│   ├── covid/
│   └── non/
└── test/
    ├── covid/
    └── non/
```

## Implementation Journey
From the outset, the integrity and quality of the dataset were validated. Image augmentation was employed to increase the dataset's size, amplifying the diversity of images for the training phase.

## Chosen Models and Rationale
I selected a suite of CNNs known for their robust performance in image classification tasks:

- MobileNetV2 (lecturer's suggestion)
- MobileNetV3 (lecturer's suggestion)
- ResNet50 (a balanced choice in the ResNet family)
- EfficientNet-B0 (offers a good trade-off between efficiency and accuracy)
- DenseNet121 (known for its efficiency in terms of parameters and depth)

## Hyperparameters and Model Configuration
Despite experimenting with various hyperparameters, it became evident that the default settings of these well-established models were most conducive to high performance. Deviating from these hyperparameters typically led to overfitting or underfitting, underscoring the sophistication of the pre-trained models.

## Performance Evaluation and Insights
The evaluation focused on the models' ability to detect COVID-19 accurately, with a particular emphasis on minimizing false negatives due to their grave implications in a medical context.

## Evaluation Metrics
Comprehensive performance metrics were employed to assess each model, with the following being a representative snapshot of the performance observed: <br>
Evaluating DenseNet121:
```
              precision    recall  f1-score   support
       covid       0.94      0.85      0.89       100
         non       1.00      1.00      1.00      3000

    accuracy                           0.99      3100
   macro avg       0.97      0.92      0.95      3100
weighted avg       0.99      0.99      0.99      3100
```

## Concluding Observations
The favorable outcomes from the validation dataset are interesting yet questionable. The inconsistent results on GradCam test on training, validation and novel images beckon a cautious approach.