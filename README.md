# ECS 171 Group 1 Project
Groupmembers: Jade Elkins, Braxton Husserl, Victor Lai, Minh Giang Tran, Chuan Hsin Wang, Martin Wong

Dataset: https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs

## Collaboration Statement
Jade Elkins: Quality Assurance

Braxton Husserl: Coding for Data Exploration and Preproccessing, Writer

Victor Lai: Coding for Data Visualization, Writer

Minh Giang Tran:

Chuan Hsin Wang: Coding for LSTM Model and string embedding, Writer

Martin Wong: Coding for Description/Company Profile Model, Writer


# Abstract
One way scammers target the vunerable is with fake job postings. Those that apply for these fake job positings are at risk of revealing their personal information to nefarious groups. An algorithm which could detect and remove these scams from job postings websites would be useful. Using a dataset which contains over 18,000 job descriptions, of which 800 are fake, we created a few machine learning models to predict whether they are fradulent. Since the data is mostly made up of natural language sentences, the data first needed to be cleaned and tokenized. This was done by setting all words to lowercase, removing punctuation and common stopwords, and then converting each string to an array of numerical values representing words. A neural network which was created and trained using only the company_profile column achieved the greatest overall accuracy of approximately 96% for both the training and test set, this also suggests that overfitting was kept to a minimum. Another model which did not remove stopwords and utilized a LSTM layer was trained on the description column and achieved a better accuracy of 97%.

# Methods
## Data Exploration
![alt text](https://github.com/bhusserl-davis/ECS171-Group1-Project/blob/main/Images/wordCloud.png?raw=true)

Most used words in the description

![alt text](https://github.com/bhusserl-davis/ECS171-Group1-Project/blob/main/Images/realvsfraudulentfrequency.png?raw=true)

Frequency of real jobs vs fraudulent jobs

We first imported the data and printed the first 5 rows of the data in order to check if we imported the correct dataset. We then check for NAN/Null values within our dataset. We see that 12 of the 18 columns contain at least 1 NaN value within it. We decided to replace the NaN values with empty strings over dropping the rows and columns with NaN. We didn’t drop any of the rows with NaN because we would be dropping at least over 15,000 of the 17,880 observations we would have. Dropping columns of our dataset would result in a shrinkage of our dataset of over 83%. Replacing the NaN values with empty strings would allow us to still utilize the columns and observations without needing to remove a majority of the data.

We then check numerical columns for unique data to see the ranges for each column and understand which columns had scaled or unscaled data. We see that telecommuting, company logo, questions, and fraudulent (the target class) have only 0 and 1 as unique values, implying they are boolean data. We can definitely use a sigmoid activation layer for our model to determine if the jobs are real or fakes.

## Data Preprocessing
All strings were converted to lowercase, that way strings like "Team" and "team" are not viewed differently by the model. While capitalization could potentially provide insight into whether the job posting is fradulent, most of the data takes the form of formal descriptions where emphasizing words with capitalization is unlikely. Next, all punctuation was replaced by whitespace, this is to further reduce noise since punctuation is not useful for our classification. Next, each string was converted to an array of strings where each element is a word, this will make language processing easier for the model. Finally, a list of stopwords were expunged from the data, this list includes common and often meaningless words for the purpose classification such as "and", "the", and "we".

It is also worth noting a bug with the dataset caused by the storage format. New line characters are not captured in our data, meaning if the original description used new lines for punctuation, such as in a bullet point list, the last word of the previous line and the first word of the next line combine into a single word (for example documentsDesign). This was mitigated in our network by separating only when there was camel casing, and excluding cases like JavaScript or SpotSource, which are product/company names. This approach minimizes the risk of falsely separating some words.

We tokenized our dataset which help make language preprocessing easier when using a neural network model. Tokenizing helps break larger phrases into smaller chunk for better analysis for sequence or patterns of words.

From the Frequency of real jobs vs fraudulent jobs, we knew that the real job's data is way more then fake jon's, so we oversample the fake job's data
```
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
```
![alt text](https://github.com/bhusserl-davis/ECS171-Group1-Project/blob/main/Images/char_after_oversample.png?raw=true)


## Neural Network On Description And Company Profile
In developing the model we used the tokenized data to create values for each word. This allows each word in the neural network to have a weight and value when being passed into the model. We first tokenize the description columns so each word within the column would have a value associated with the word. For example, the word the would have a value of 1 so any the in the column would be replaced with 1. We then pad each row of the column with a certain length of 500 to ensure each row has the same length when passed as inputs to the neural network. We would then split the split the description data into training and testing data where 20 percent of the data would be allocated to testing. The last 80 would be used for training the model. We would do the same preprocessing method for company profile column. We would tokenize and assign values to each unique word while padding the length of each row size to be 500. The data would similarly be split to the description data in a 80 to 20 ratio.

The next step was building a model. We started the first model to be relatively small and simple. The model would have 3 layers where the first layer was a tanh activation function with 500 nodes. The second layer would be a relu activation function with 100 nodes. We would then have the third layer be a sigmoid layer where the values would be converted between 0 and 1. 

The model would then be fitted using the description data and the company profile data. The two models would then predict the class based on the x_test adata and the outputted result would be converted to binary values with a threshold of 0.5 because the y_test consists of binary values. We then printed the accuracy and classification report for description and company profile data.

## LSTM Model With different way to encode the string
- In this model we elected not to remove common stopwords such as "and", "the", and "to".
### 1. only tokenized the string
### 2. one-hot encoding
- convert each word into one-hot encoding, got similer outcome as only tokenized the string
### 3. word2Vec
- the another way to convert the string is using Word2Vec, which will convert each word to a vector. The words has similar meaning will near to each other in the space, and the vector's direction will show the relationship of words
```
# initialize & training our word2vec model
w2v_model = gensim.models.Word2Vec(sentence,min_count=1,size=100)
```
- because we are going to add embedding layer in the model, so we need to create a convert matrix which will use in embedding layer
- to create the convert matrix, we need to first using tokenizer to convert each word to a number (ex. and -> 1, the -> 2, to -> 3.etc)
- and the `get_weight_matrix` function will create a matrix which map 1->the vector which  word "and" being convert by word2vec mode
```
# mapped the word's index (by tokenizer) to the word's vector (by word2vec model)
def get_weight_matrix(model, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix

```
### 4. TFIDF
- TFIDF is mean how important this word is in the text, so it will convert each word into a value


# Results
We would see that the description yielded a 95.7% accuracy while the company profile yielded a 93.5%

# Discussion
The description and company profile neural network models have a very low recall for fraudulent predictions, this implies that their high accuracy is a result of prediciting non-fradulent nearly every time. In fact, these models correctly predicted fradulent with an accuracy of less than 10%. This could be due to the comparitvely small amount of fraudulent entries there are to train on since the dataset consists of 95% true job entries. 

Using Word2Vec to do string embedding achieved a much higher recall for fradulent entires, because word2Vec will consider about the word's meaning to encode the word, so we can know that Word2Vec is a better way to encode. 

After oversample the fradulent entires, we can get 1.0 recall and 0.99 accuracy, also the one only tokenized the string can get much higher recall, so that the most important thing is to balance our datasets. 

|  LSTM model      | only tokenized | using Word2Vec | only tokenized + oversample | Word2Vec + oversample |
| ---------------  | -------------- | -------------- | --------------------------- | --------------------- |
| recall           | 0.1            | 0.5            | 0.97                        | 1.0                   |
| accuracy         | 0.95           | 0.97           | 0.89                        | 0.99                  |

# Conclusion
We were able to achieve great accuracy with no noticable overfitting while only taking into account the job description, this suprised me since I cannot decern between the fradulent jobs myself. It wasn't untill after investigating further that I realized why, since our dataset is over 95% non-fraudulent entries, the model could simply predict non-fraudulent in every case and achieve a 95% accuracy. In fact, that is exactly what it did, seeing as the description based model only classified a single observation as fraudulent in the entire test set. The question of course is, how can we fix this problem? The second model which utlized a LSTM layer and did not remove the most common words such as "and", "the", and "to" had a much higher success for predicting fraudulent entries, implying that these common words are actually very important. The ratio of fraudulent to non-fraudulent entries in the dataset also greatly contributed to the model's bias. If this problem is tackled again, two things to keep in mind are to utilize a much more balanced dataset, either with more fraudulent examples or with a more even ratio, and to keep in common stopwords from the data.
