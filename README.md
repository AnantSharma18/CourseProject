
# CourseProject | Text and Tweet Classification using Machine Learning

This project is focussed at classifying texts into their relevant categories using different machine learning techniques. Text classification is one of the standard applications in text mining. . The objective of our text classification task is to find appropriate labels for previously unlabelled data from a predictive model which has been trained on a pre-labelled dataset. A series of necessary subtasks are performed to identify and extract relevant features from a given text, which can be further applied to train a predictive model.

The following Classifications have been accomplished in this project:
1. Classify SMS into Spam or Not Spam  
2. Classify text into different hate-speech and offensive language category 
3. Classify political tweets (Indian context) using a custom dataset

## Datasets
The details for the datasets is as follows:
 - **SMS Spam Collection**
Source: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/  
The SMS Spam Collection v.1 is a public set of SMS labeled messages that have been collected for mobile phone spam research. It has one collection composed by 5,574 English, real and non-enconded messages, tagged according being legitimate (ham) or spam.  
`Labels : spam / ham`

 
 - **Hate-speech and Offensive Language**
Source: Dataset https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data  
The data are stored as a CSV and as a pickled pandas dataframe (Python 2.7). Each data file contains 5 columns:
`count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF). hate_speech = number of CF users who judged the tweet to be hate speech. offensive_language = number of CF users who judged the tweet to be offensive. neither = number of CF users who judged the tweet to be neither offensive nor non-class = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 – neither Labels : hate speech / offensive language / neither
`
-   **Political Tweets Dataset – Custom | Indian Context**
The political tweets have been taken from a GitHub repo for Sentiment Analysis on Indian Political fetched tweets.
Source: https://github.com/rohitgupta42/polity_senti
Further, a set of General tweets was taken from the The Twitter Political Corpus
Source: https://www.usna.edu/Users/cs/nchamber/data/twitter/
Both of these datasets were combined and cleaned to obtain the required datasets.
`Labels : POL / NOTPOL`

## Methodology

In this project, I have implemented multiple models in order to classify textual data from publicly available datasets initially. Each model has been implemented in a seperate Jupyter Notebook for clear understanding. A comparison study is also presented between the various models in the result section of this documentation.

**Pre-processing**
Cleaning of text  
Stop Words removal  
Stemming  
Removal of non-alphabetic characters

**Feature Extraction**
Count Vectors b. TF-IDF Vectors
Bag of words (word level)
Bag of n-grams (n-gram level)
Character level
    
### Implemented Models
1.  Naïve Bayes
2.  Linear Classification
3.  SVM
4.  Random Forest
5.  Convolutional Neural Network
    
At the end of each model notebook, an accuracy report is present. The classification accuracy of different models is summarised in the results section below.

## Results

The following table contain the F1 Scores for each of the predicted categories:

### Naive Bayes
| Model / Feature | SMS Spam | Offensive language | Political Tweet |
|-----------------|:--------:|:------------------:|:-----------------:|
| Count           |   **0.88**   |        **0.96**        |       **0.97**      |
| Word TF-IDF     |   0.83   |        0.93        |       0.92      |
| N-Gram TF-IDF   |   0.60   |        0.91        |       0.80      |
| Char TF-IDF     |   0.71   |        0.94        |       0.91      |

### Linear Classification
| Feature / Cagtegory | SMS Spam | Offensive language | Political Tweet |
|-----------------|:--------:|:------------------:|:-----------------:|
| Count           |   **0.91**   |        0.97        |       0.96      |
| Word TF-IDF     |   0.81   |        **0.97**        |       **0.97**      |
| N-Gram TF-IDF   |   0.18   |        0.91        |       0.82      |
| Char TF-IDF     |   0.84   |        0.96        |       0.95      |

### SVM
| Model / Feature | SMS Spam | Offensive language | Political Tweet |
|-----------------|:--------:|:------------------:|:-----------------:|
| Count           |   0.89   |        0.97        |       0.95      |
| Word TF-IDF     |   **0.92**   |        **0.97**        |       **0.96**      |
| N-Gram TF-IDF   |   0.79   |        0.92        |       0.81      |
| Char TF-IDF     |   0.88   |        0.97        |       0.94      |

### Random Forest
| Model / Feature | SMS Spam | Offensive language | Political Tweet |
|-----------------|:--------:|:------------------:|:---------------:|
| Count           |   **0.86**   |        0.97        |       0.95      |
| Word TF-IDF     |   0.85   |        **0.97**        |       **0.96**      |
| N-Gram TF-IDF   |   0.74   |        0.92        |       0.81      |
| Char TF-IDF     |   0.83   |        0.96        |       0.94      |

### Convolutional Neural Network (CNN)
| CNN / Category | Spam SMS | Offensive Lang | Political Tweet |
|---------------------|:--------:|:--------------:|:---------------:|
| F1 Score            |   0.91   |      0.97      |       0.95      |
| Accuracy            |   0.98   |       0.95         |       0.94      |

### Summarized Results
The best performing features have the following F1 Scores in each of the models:
| Model / Category      | Spam SMS | Offensive Lang | Political Tweet |
|-----------------------|:--------:|:--------------:|:---------------:|
| Naive Bayes           |   0.88   |      0.96      |       0.95      |
| Linear Classification |   0.90   |      0.97      |       0.96      |
| SVM                   |   0.92   |      0.97      |       0.96      |
| Random Forest         |   0.86   |      0.97      |       0.96      |
| CNN                   |   0.91   |      0.96      |       0.95      |

## Political Tweets Prediction
The `Political Tweet Prediction` notebook can be used to fetch and predict tweets into Political and Non Political Categories. The tweets are fetched using the tweepy package which uses the Twitter API to stream tweets pertaining to a certain tracking list. 

The most accurate model for political tweet prediction i.e. Linear Classification with Word Level TF-IDF features has been saved as a pickle model is used to categorise the tweets in pol / notpol categories.

## Installation Guide

The project makes use of Jupyter Notebook to implement the different models. The following pre-requisites are required to run the code
Python3 
Libraries
