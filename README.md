# SPORTS IMAGE CLASSIFICATION USING TENSORFLOW
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

## PROJECT OVERVIEW
---------------------------------------------------------------------------------

This repository is a project dedicated to classifying various sports images to their respective sports using __`Convolutional Neural Networks`__.

We use __`Transfer Learning`__ to utilise prebuilt models on __`imagenet`__ dataset and try to improve on their performance using fine-tuning techniques.

## ABOUT THE DATASET
---------------------------------------------------------------------------------

The dataset used for the project, contains images from 100 different kinds of sports and is already split into train, valid and test sets to make life a bit easier. The dataset has been copied from a __`kaggle`__ source.

Link to the dataset: (https://www.kaggle.com/datasets/gpiosenka/sports-classification?select=sports.csv)

Contains: 13493 train, 500 test, 500 validate images.

## HOW TO USE
----------------------------------------------------------------------------------

First and foremost, clone the repository to your local using:

`git clone https://github.com/abhijitchak103/sports-classification-cnn.git`

To download the dataset, you can download the zip file and unzip manually to `/data` folder in your local copy of this repo or use the Terminal to help you with it.

First of all, make sure you have copied your kaggle key from account to your local directory in the folder: 
`C:\Users\{user}\.kaggle`

Then in the command line

### USE LOCALLY

#### Without Docker

```python
ipython
import lambda_function
url = event[$url] # $url=url-to-check
lambda_function.lambda_handler(event, None)
```
OR
```python
ipython
import lambda_function
lambda_function.predict($url) # $url=url-to-check
```
#### With Docker
