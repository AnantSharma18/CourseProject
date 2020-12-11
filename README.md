
# CourseProject | Text and Tweet Classification using Machine Learning

This project is focussed at classifying texts into their relevant categories using different machine learning techniques. Text classification is one of the standard applications in text mining. . The objective of our text classification task is to find appropriate labels for previously unlabelled data from a predictive model which has been trained on a pre-labelled dataset. A series of necessary subtasks are performed to identify and extract relevant features from a given text, which can be further applied to train a predictive model.

The following Classifications have been accomplished in this project:
1. Classify SMS into Spam or Not Spam  
2. Classify text into different hate-speech and offensive language category 
3. Classify political tweets (Indian context) using a custom dataset

The details for the datasets is as follows:

 - SMS Spam Collection
Source: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/  
The SMS Spam Collection v.1 is a public set of SMS labeled messages that have been collected for mobile phone spam research. It has one collection composed by 5,574 English, real and non-enconded messages, tagged according being legitimate (ham) or spam.  
`Labels : spam / ham`

 
 - Hate-speech and Offensive Language
Source: Dataset https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data  
The data are stored as a CSV and as a pickled pandas dataframe (Python 2.7). Each data file contains 5 columns:
`count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF). hate_speech = number of CF users who judged the tweet to be hate speech. offensive_language = number of CF users who judged the tweet to be offensive. neither = number of CF users who judged the tweet to be neither offensive nor non-class = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 – neither Labels : hate speech / offensive language / neither
`
-   Political Tweets Dataset – Custom | Indian Context
The political tweets have been taken from a GitHub repo for Sentiment Analysis on Indian Political fetched tweets.
Source: https://github.com/rohitgupta42/polity_senti
Further, a set of General tweets was taken from the The Twitter Political Corpus
Source: https://www.usna.edu/Users/cs/nchamber/data/twitter/
Both of these datasets were combined and cleaned to obtain the required datasets.