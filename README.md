# AI-based-low-cost-pulse-oximeter
Pulse oximetry is a noninvasive test that measures the oxygen saturation level of the blood. Oxygen saturation (SPO2) is a measure of how much oxygen the blood is carrying as the percentage of the maximum it could carry. For a normal person it is in the range of 95-100%. Furthermore, it could increase and drop due to many factors.

This project covers the making of a basic and low cost pulse oximeter to measure SPO2 as well as give a prediction on the state of the SPO2 levels. It is implemented on hardware and the results are used for training a classification algorithm which predicts 9 different kinds of diseases related to oxygen levels and heart beats per minute. In the following sections, this report will discuss each aspect of the implementation.

## Hardware Implementation

### Block DIagram and Circuit Diagram
![image](https://github.com/user-attachments/assets/87007183-2f2b-4170-b67a-9de88d600060)
![image](https://github.com/user-attachments/assets/c7b95d5a-26b5-4757-a74e-6d168e065828)

## Software Implementation
### Dataset
#### Data
The data was generated using built-in “randint” function of python. It was done so because we could not find any dataset which would be diverse in terms of the different conditions associated with the different levels of SPO2 combined with different BPM (heart beats per minute).
Basically the data was generated into 3 ranges for SPO2 and BPM as explained in table 1. This random data ranged was combined into 1 array for SPO2 and BPM each.
#### Preprocessing
Class labels were assigned to the dataset based upon the conditions of the SPO2 and BPM as given in table 1. This gave us 9 classes from 0-8.
Finally the data frame was saved into a csv file for future use. To make the data uniform, normalization was performed between the range of 0 and 1.
![image](https://github.com/user-attachments/assets/e14eb0b8-ef3e-4cda-88a8-274c64e81f7d)

### Classification model
As models do not accept data frames, it was converted back to array. To proceed further, the data was split into 2 categories of train and test with ratio of 70-20. It was also done to prevent model from overfitting and cramming instead of learning features.
As our data has 9 classes, a 5 layered neural network model was trained. The model has 3 hidden layers with 100,50,9 nodes. After specifying the model specifications, the model was fit with 50 epochs and a validation set of 0.1 of train data.
#### Activation function
The activation function given to the first 2 hidden layers is relu and for last hidden layer is softmax. Relu is fast and is computationally very efficient. The Softmax function is used for prediction in multi-class models where it returns probabilities of each class in a group of different classes, with the target class having the highest probability. 
##### Loss function
Sparse categorical cross entropy proves a flexibility in terms of the input variables. This loss computes logarithm only for output index which ground truth indicates to.

## Project pipeline
The project was implemented in two steps:
i. Oximeter circuit was first implemented on hardware as described in the previous section.
ii. A classification model was trained using tensorflow 5 layers neural network classifier.
iii. Real-time readings from oximeter were given as input to the classification model to predict classes.

## Model Architecture 
![image](https://github.com/user-attachments/assets/dca1f10c-c9ec-4b2f-a1c3-dba2efdd5f09)

## Results
### Hardware implementation
![image](https://github.com/user-attachments/assets/85d0e065-14fa-4f1f-87e4-d24d43f375cc)
![image](https://github.com/user-attachments/assets/d7d731a8-cf88-4355-8ff5-d09cf32db7ee)

### Classification Algorithm
#### Training
Training accuracy:
![image](https://github.com/user-attachments/assets/6bfc3f94-c372-4c59-ac13-98fd9a7d1931)

Training loss:
![image](https://github.com/user-attachments/assets/ce83f48e-7c2a-4bef-8961-c28083830529)

#### Test set
Confusion matrix:
![image](https://github.com/user-attachments/assets/35eed6e0-5b7e-4a49-8914-e4f1de4e02a5)

Prediction on arduino input:
![image](https://github.com/user-attachments/assets/372db8f7-9670-4c23-8e5e-260b9eb94106)

## Conclusion
A low-cost oximeter was successfully implemented using Arduino and MAX30100 module. Since there was no dataset available, it was randomly generated for SPO2 and BPM, keeping in mind the ranges for a normal, healthy human. The data was labelled as a class from 0 – 8, indicating one of the following diseases: normal, severe cyanosis, hypoxia, dead, choke, cyanosis, tachycardia, severe hypoxia, arrhythmia.
A 5-layer neural network was trained for these classes. The training accuracy was achieved at 89%. After training the model, it was saved. Arduino output was given to the saved model and used to make predictions in real time.


