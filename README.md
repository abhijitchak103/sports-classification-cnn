# SPORTS IMAGE CLASSIFICATION USING TENSORFLOW

## PROJECT OVERVIEW

This repository is a project dedicated to classifying various sports images to their respective sports using __`Convolutional Neural Networks`__.

We use __`Transfer Learning`__ to utilise prebuilt models on __`imagenet`__ dataset and try to improve on their performance using fine-tuning techniques.

## ABOUT THE DATASET

The dataset used for the project, contains images from 100 different kinds of sports and is already split into train, valid and test sets to make life a bit easier. The dataset has been copied from a __`kaggle`__ source.

Link to the dataset: (https://www.kaggle.com/datasets/gpiosenka/sports-classification?select=sports.csv)

Contains: 13493 train, 500 test, 500 validate images.

## HOW TO USE

First and foremost, clone the repository to your local using:

`git clone https://github.com/abhijitchak103/sports-classification-cnn.git`

To download the dataset, you can download the zip file and unzip manually to `/data` folder in your local copy of this repo or use the Terminal to help you with it.

First of all, make sure you have copied your kaggle key from account to your local directory in the folder: 
`C:\Users\{user}\.kaggle`

Then change to the directory where you cloned the repo using `cd`.
```
kaggle datasets download -d gpiosenka/sports-classification
unzip sports-classification.zip -d data
rm sports-classification.zip
cd data
rm "EfficientNetB0-100-(224 X 224)- 98.40.h5"
cd ..
```

This should unzip the required data used in this project to the correct folder structure.

After this, the folder structure would look like this:
```
.
├── data
│   ├── test
│   ├── train
│   ├── valid
│   └── sports.csv
├── Dockerfile
├── README.md
├── .gitignore
├── lambda_function.py
├── models
│   ├── mobilenetv2_v5_aug_height_16_0.922.h5
│   ├── mobilenetv2_v6_aug_rot_17_0.920.h5
│   ├── mobilenetv2_v7_17_0.912.h5
│   └── prediction.tflite
├── notebooks
│   ├── model-converter.ipynb
│   ├── sports-classification.ipynb
│   └── test.ipynb
├── requirements.txt
├── test.py
└── utils.py
```

### CREATE VIRTUAL ENVIRONMENT

To run the project and to test out notebooks, you can create a new virtual environment using a framework of your choice. eg.
`
conda create -n project python==3.8 -y
conda activate project
`
Once you activate the venv, install the dependencies:
`
pip install -r requirements.txt
`

### TEST NOTEBOOKS

To test out the notebooks and rerun the entire thing, you can run all. Keep in mind, this is will be time-taking. 
To do the same:

`
jupyter notebook
`
In the web browser, the jupyter notebook environment should be open. Open the notebooks folder, and open `sports-classification.ipynb` to test it.

### MAKE PREDICTIONS LOCALLY

To test the notebook locally you can use docker or run without docker.

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
This should give the following output with url: 
'https://images.pexels.com/photos/3628912/pexels-photo-3628912.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1'

`
[('cricket', 10.8796835),
 ('baseball', 8.193538),
 ('croquet', 7.908885),
 ('football', 6.1987066),
 ('golf', 5.5249014)]
`

#### With Docker

`
docker build -t sports .
docker run -it --rm -p 8080:8080 sports
`
Then in a new terminal, cd to the working directory.
`
conda activate project
python test.py
`
This should give you the following result:
`['cricket', 10.8796835]`

## CONCLUSION

The model yields an accuracy of 93% on test set, which is lower than the benchmark 95%. 

## NEXT STEPS

- Fine tune and test with different pre-built models to improve model accuracy.
- Build a model from scratch without using Transfer Learning

## CONTRIBUTORS

Abhijit Chakraborty (ab.chakraborty@outlook.com)

## ACKNOWLEDGEMENTS

- [Alexey Grigorev](https://github.com/alexeygrigorev)
- [DataTalks.Club](https://datatalks.club/)

## CONTRIBUTIONS

All sorts of contributions and ideas are welcome to add on to the current project and improve the models. Any feedback received will be highly appreciated.

