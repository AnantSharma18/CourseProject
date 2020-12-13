
# CourseProject | Text and Tweet Classification using Machine Learning

This project is focussed at classifying texts into their relevant categories using different machine learning techniques. Text classification is one of the standard applications in text mining. . The objective of our text classification task is to find appropriate labels for previously unlabelled data from a predictive model which has been trained on a pre-labelled dataset. A series of necessary subtasks are performed to identify and extract relevant features from a given text, which can be further applied to train a predictive model.

The following Classifications have been accomplished in this project:
1. Classify SMS into Spam or Not Spam  
2. Classify text into different hate-speech and offensive language category 
3. Classify political tweets (Indian context) using a custom dataset

> Note : The purpose of this project is to present the different machine
> learning implementations to classify textual data. While the
> accuracies of the different models are mentioned in the result
> section, the project focussed more on the different ways machine
> learning can be used to perform text classification.

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

The project makes use of Jupyter Notebook to implement the different models. The following steps to install and run the code on your local machine.

### Running the Helper Notebooks

1. Clone this repo onto your local machine.
2. You will need the following pre-requisites installed in your local environment: `python3, jupyter notebook, numpy, pandas, keras, tensorflow, sklearn, spacy, ntlk, string, pickle, twitter`
3. Once you have made sure that the above mentioned packages are installed in you environment, you can go ahead and launch jupyter notebook
4. Each Helper Notebook can be executed seperately to get results for different models.
5. In each of the Helper Notebook, the variable `currentDF` can be changed to either `OffensiveLangDF | spamSmsDF | politicalDF` to run the model on different datasets


Note : You might have to download the spacy 'en' model seperately in order to run the notebooks

### Predicting Political Tweets

1. The pre-trained model to predict political tweets is already saved in the folder 'Saved Model' under the name `LR_Pol.plk`
2. In order to run the political tweet prediction notebook, you will have to first obtain credentials from Twitter API. 
3. The `consumer_key, consumer_secret, access_token` and `access_token_secret` variables need to be replaced with your unique Twitter API credentials. 
4. Guide to obtain Twitter API credentials can be found under references.
5. Once you have replaced the `XXXX` with your Twitter API credentials, you can execute the notebook to obtain 4 different tweets using the tracking list and classfiy them into pol / notpol using the pre-trained model
6. The `trackingList1` and `trackingList2` lists can be edited to stream different tweets.
7. The `n_tweets` variable can be changed to the number of tweets you wish to obtain.
	
## Contribution
Project by `Anant Ashutosh Sharma`
Free Topic : `Text and Tweet Classification using Machine Learning`
Course : `CS 410`
NetID : `anantas2`

The following documentation is submitted to the GitHub Repo

       1. Project Proposal
       2. Project Progress Report
       3. Self-Evaluation Report 

## Limitations
The datasets being used to train the model can be further improved. They currently have very niche examples of the catergories. For instance, the political tweets dataset only contains political tweets pertaining to the Indian Context. Further, the number of training records are less. A higher number of traininf records may allow the models to perform better.

The Convulutional Neural Network (CNN) implemented in this project is a simple and generic one. A much more complex and accurate CNN can be designed and fine tuned as per the requirements of each of the datasets. 

These limitations are present in this project since the purpose of this project is not to present any accurate model to classify text objects, but to present the different methods and ways in which machine learning models can be used to classify textual data. 

## References

 - https://rapidapi.com/blog/how-to-use-the-twitter-api/
 - http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
 - https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data  
 - https://github.com/rohitgupta42/polity_senti
 - https://www.usna.edu/Users/cs/nchamber/data/twitter/
 - scionoftech GitHub Repo for helping in eaxtracting different features for models

