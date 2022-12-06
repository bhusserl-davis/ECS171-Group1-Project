# ECS 171 Group 1 Project
Groupmembers: Jade Elkins, Braxton Husserl, Victor Lai, Minh Giang Tran, Chuan Hsin Wang, Martin Wong

Dataset: https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs

## Collaboration Statement
Jade Elkins: Coding for Data Preprocessing, Quality Assurance, Writer

Braxton Husserl: Coding for Data Exploration and Preproccessing, Writer

Victor Lai: Coding for Data Visualization, Writer

Minh Giang Tran: Coding for Data Visualization

Chuan Hsin Wang: Coding for LSTM Model and string embedding, Writer

Martin Wong: Coding for Description/Company Profile Model, Clean Nan, Writer


# Abstract
One way scammers target the vunerable is with fake job postings. Those that apply for these fake job positings are at risk of revealing their personal information to nefarious groups. An algorithm which could detect and remove these scams from job postings websites would be useful. Using a dataset which contains over 18,000 job descriptions, of which 800 are fake, we created a few machine learning models to predict whether they are fradulent. Since the data is mostly made up of natural language sentences, the data first needed to be cleaned and tokenized. This was done by setting all words to lowercase, removing punctuation and common stopwords, and then converting each string to an array of numerical values representing words. The first two model which were trained using only the description and company_profile column respectively each achieved an overall accuracy of 96% but with a very low recall of less than 10% for the fradulent class. Another model which did not remove stopwords and used Word2Vec and a LSTM layer achieved a better accuracy of 97% and recall of 50%. This model when combined with undersampling achieved a 93% accuracy but with a 97% recall on the fraudulent class. The final and best model which utilized oversampling, Word2Vec, and LTSM achieved an overall accuracy of 98% with 60% recall on the fradulent class.

# Methods
## Data Exploration
![alt text](https://github.com/bhusserl-davis/ECS171-Group1-Project/blob/main/Images/wordCloud.png?raw=true)

Most used words in the description

![alt text](https://github.com/bhusserl-davis/ECS171-Group1-Project/blob/main/Images/realvsfraudulentfrequency.png?raw=true)

Frequency of real jobs vs fraudulent jobs

We first imported the data and printed the first 5 rows of the data in order to check if we imported the correct dataset. We then check for NaN/Null values within our dataset. We see that 12 of the 18 columns contain at least 1 NaN value within it. We decided to replace the NaN values with empty strings over dropping the rows and columns with NaN. We didn’t drop any of the rows with NaN because at least over 15,000 of the 17,880 observations include NaN data. Dropping columns of our dataset would result in a shrinkage of our dataset of over 83%. Replacing the NaN values with empty strings would allow us to still utilize the columns and observations without needing to remove a majority of the data.

We then check numerical columns for unique data to see the ranges for each column and understand which columns had scaled or unscaled data. We see that telecommuting, company logo, questions, and fraudulent (the target class) have only 0 and 1 as unique values, implying they are boolean data. The rest were included natural language data. We can definitely use a sigmoid activation layer for our model to determine if the jobs are real or fakes.

## Data Preprocessing
All strings were converted to lowercase, that way strings like "Team" and "team" are not viewed differently by the model. While capitalization could potentially provide insight into whether the job posting is fradulent, most of the data takes the form of formal descriptions where emphasizing words with capitalization is unlikely. Next, all punctuation was replaced by whitespace, this is to further reduce noise since punctuation is not useful for our classification. Next, each string was converted to an array of strings where each element is a word, this will make language processing easier for the model. Finally, a list of stopwords were expunged from the data, this list includes common and often meaningless words for the purpose classification such as "and", "the", and "we".

It is also worth noting a bug with the dataset caused by the storage format. New line characters are not captured in our data, meaning if the original description used new lines for punctuation, such as in a bullet point list, the last word of the previous line and the first word of the next line combine into a single word (for example documentsDesign). This was mitigated in our network by separating only when there was camel casing, and excluding cases like JavaScript or SpotSource, which are product/company names. This approach minimizes the risk of falsely separating some words.

We tokenized our dataset which help make language preprocessing easier when using a neural network model. Tokenizing helps break larger phrases into smaller chunk for better analysis for sequence or patterns of words.

From the frequency of real job entries vs fraudulent job entries, we knew that there were many more real entries than fake entries, so we oversample the fake job's data
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

## LSTM Model With word2Vec to encode the string
### Word2Vec encoding
ch will convert each word to a vector. The words has similar meaning will near to each other in the space, and the vector's direction will show the relationship of words
```
# initialize & training our word2vec model
w2v_model = gensim.models.Word2Vec(sentence,min_count=1,size=100)
```

### RNN v.s. LSTM
When we understand the meaning of a sentence, to understand each word of the sentence in isolation is not enough, we need to deal with the whole context and how those word connected. Normal NN can only deal with each input at a time. RNN will base on the new information and last time's outcome to generate new outcome, but it cannot remember the information that is too long ago.
LTSM is a special kind of Recurrent Neural Network(RNN), LTSM will learn what information need to save, and what should forget.
```
#Defining Neural Network
model = Sequential()
#Non-trainable embeddidng layer
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))
#LSTM 
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
```
The first layer of the model is embedding layer, we create a convert matrix,  to covert tokennize to vector( we already use tokenizer to convert each word to a number (ex. and -> 1, the -> 2, to -> 3.etc). 
The `get_weight_matrix` function will create a matrix which map 1->the vector which  word "and" being convert by word2vec model.
```
# mapped the word's index (by tokenizer) to the word's vector (by word2vec model)
def get_weight_matrix(model, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix
 
embedding_vectors = get_weight_matrix(w2v_model, word_index)
```
In the second layer, we tried RNN layer and LTSM, in both model we use 128 units.
In the third layer, we use sigmoid as our activation function.


# Results
In the table below are the results from our five main models. For each, the recall of the non-fraudulent (0) class and fraudulent (1) class as well as overall accuracy are listed.

|  Metric   | only tokenized (Description) | only tokenized (company_profile) | oversample + Word2Vec | oversample + Word2Vec + RNN | oversample + Word2Vec + LTSM |
| ------     | ---------------------------- | ------------------------------- | --------------------- | --------------------------- | --------------------------- |
| recall (0) | 1.00                         | 1.00                            | 0.93                  | 0.62                        | 0.66 |
| recall (1) | 0.00                         | 0.04                            | 0.51                  | 0.97                        | 0.98 |
| accuracy   | 0.96                         | 0.96                            | 0.92                  | 0.98                        | 0.99 |


# Discussion
The description and company profile neural network models have a very low recall for fraudulent predictions, this implies that their high accuracy is a result of prediciting non-fradulent nearly every time. In fact, these models correctly predicted fradulent entries with less than 10% success. This could be due to the comparatively small amount of fraudulent entries there are to train on since the dataset consists of 95% true job entries. 

Using Word2Vec to do string embedding achieved a much higher recall for fradulent entires, because word2Vec takes into account the word's meaning to encode the word, this shows that Word2Vec is a better way to encode. 

After oversampling the fradulent entires, we can get 0.6 recall on the fradulent class and 0.98 overall accuracy. The model which only tokenized the string and oversampled (no Word2Vec) also got a high recall, this implies that the most important thing is to balance our datasets. 

|  LSTM model      | only tokenized | using Word2Vec | only tokenized + oversample | Word2Vec + oversample |
| ---------------  | -------------- | -------------- | --------------------------- | --------------------- |
| recall (1)       | 0.01           | 0.5            | 0.51                        | 0.6                   |
| accuracy         | 0.95           | 0.97           | 0.89                        | 0.98                  |

We can also choose to undersample the non-fradulent data, we remove the non-fradulent data observations with NaN value first before we oversample it. Initally it seens to produce poor accuracy, but after more epochs it also achieves higher accuracy and recall.
![alt text](https://github.com/bhusserl-davis/ECS171-Group1-Project/blob/main/Images/char_after_undersample.png?raw=true)

|  LSTM model + remove Nan observation     | only tokenized + 6 epoches | only tokenized  + 60 epoches | using Word2Vec + 6 epoches | Word2Vec  + 60 epoches |
| ---------------------------------------  | -------------------------- | -------------------------- | ---------------------------- | ---------------------- |
| recall (0)                               | 0.50                       | 0.79                       | 0.77                         | 0.79                   |
| recall (1)                               | 0.78                       | 0.81                       | 0.71                         | 0.76                   |
| accuracy                                 | 0.66                       | 0.80                       | 0.74                         | 0.77                   |

This shows that the amount of training data is also important, but if we don't have enough data, we can use more training epochs to make up for it.

# Conclusion
We were able to achieve great accuracy with no noticable overfitting while only taking into account the job description, this suprised me since I cannot decern between the fradulent jobs myself. It wasn't untill after investigating further that I realized why, since our dataset is over 95% non-fraudulent entries, the model could simply predict non-fraudulent in every case and achieve a 95% accuracy. In fact, that is exactly what it did, seeing as the description based model only classified a single observation as fraudulent in the entire test set. The question of course is, how can we fix this problem? 

The model which utlized a LSTM layer and word2Vec to enocode data had a much higher success for predicting fraudulent entries, implying the way data is encoded is actually very important. The ratio of fraudulent to non-fraudulent entries in the dataset also greatly contributed to the model's bias. After we oversampled the fraudulent data to make our dataset more balanced, the recall was significantly improved. This result shows that the balance of the data is very important.

# reference
- https://www.kaggle.com/code/kumbaraci/text-classification
- https://www.kaggle.com/code/atishadhikari/fake-news-cleaning-word2vec-lstm-99-accuracy
