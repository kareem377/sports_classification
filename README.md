# Code Description

## Overview
This code performs classification on a sports image dataset using Convolutional Neural Networks (CNNs) and the VGG architecture. It uses the TensorFlow and Keras libraries for building and training the models. The code includes data preprocessing, model building, training, evaluation, and prediction functionalities.

## Dependencies
The code requires the following libraries to be installed:
- pandas
- numpy
- os
- matplotlib
- seaborn
- warnings
- random
- tensorflow
- sklearn
- PIL
- tqdm

## Usage
1. Make sure all the required libraries are installed.
2. Mount the Google Drive using `drive.mount('/content/drive/')` to access the dataset stored in Google Drive.
3. Specify the paths of the training and testing datasets in the `Trainingset` and `Testingset` variables, respectively.
4. Define the `data()` function to extract image paths and corresponding labels from the dataset directories.
5. Load and preprocess the training and testing data using the `data()` function and the `featureExtruction()` function.
6. Split the data into features (input images) and labels (target classes) and perform label encoding and one-hot encoding.
7. Build the CNN model using the `Sequential` API and add convolutional, pooling, flatten, and dense layers.
8. Compile the CNN model with an optimizer, loss function, and metrics.
9. Train the CNN model using the training data and evaluate it using the testing data.
10. Save the trained model for future use.
11. Build the VGG model using the `Sequential` API and add convolutional, pooling, dropout, flatten, and dense layers.
12. Compile the VGG model with an optimizer, loss function, and metrics.
13. Train the VGG model using the training data and evaluate it using the testing data.
14. Save the trained VGG model for future use.
15. Use the `prediction()` function to predict the class of a single image by providing its path.
16. Use the `pred_from_test()` function to predict the classes of multiple images from the testing dataset.
17. The predictions will be displayed along with the corresponding images.

## File Structure
- The code assumes that the dataset is stored in Google Drive under the specified paths.
- The code saves the trained models in the Google Drive directory.
- The code generates prediction results in CSV format.

## Note
Please note that the code assumes the availability of the dataset and the correct file paths. Make sure to adjust the paths and directory structure according to your specific dataset setup.

## Contact
If you have any questions or need further assistance, please feel free to contact [kareemhatem37@gmail.com].
