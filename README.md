# ECS 171 Group 1 Project

Groupmembers: Jade Elkins, Braxton Husserl, Victor Lai, Minh Giang Tran, Chuan Hsin Wang, Martin Wong
Dataset: https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs

# Abstract
### One way scammers target the vunerable is with fake job postings. Those that accept these fake job positings are at risk of revealing their personal information to nefarious groups. An algorithm which could detect and remove these scams from job postings websites would be useful. Using a dataset which contains over 18,000 job descriptions, of which 800 are fake, we created a few machine learning models to predict whether they are fradulent. Since the data is mostly made up of natural language sentences, the data first needed to be cleaned and tokenized. This was done by setting all words to lowercase, removing punctuation and common shortwords, and then converting each string to an array of numerical values representing words. A neural network which was created and trained using only the company_profile column achieved the greatest overall accuracy of approximately 96% for both the training and test set, this also suggests that overfitting was kept to a minimum. 

# Background
### Being able to differentiate between real and fake job postings would decrease the risk of applying for scam positions, leaving more time to apply for real, potential job offerings. The dataset contains 18k job descriptions, 800 of which are fake. It contains eighteen different features including job title, job description, salary, industry, and whether they are fraudulent or not. This excercise will give us a greater understanding of how to build models that work on natural language data.

# Introduction
### We are planning on using a natural language processing scheme such as a recurrent neural network or transformer to predict whether the job listing is fraudulent or not.

## Data Exploration
### We first imported the data and printed the first 5 rows of the data in order to check if we imported the correct dataset. We then check for NAN/Null values within our dataset. We see that 12 of the 18 columns contain at least 1 NaN value within it. We decided to replace the NaN values with empty strings over dropping the rows and columns with NaN. We didn’t drop any of the rows with NaN because we would be dropping at least over 15,000 of the 17,880 observations we would have. Dropping columns of our dataset would result in a shrinkage of our dataset of over 83%. Replacing the NaN values with empty strings would allow us to still utilize the columns and observations without needing to remove a majority of the data.

### We then check numerical columns for unique data to see the ranges for each column and understand which columns had scaled or unscaled data. We see that telecommuting, company logo, questions, and fraudulent (the target class) have only 0 and 1 as unique values, implying they are boolean data. We can definitely use a sigmoid activation layer for our model to determine if the jobs are real or fakes.

## Data Preprocessing
### All strings were converted to lowercase, that way strings like "Team" and "team" are not viewed differently by the model. While capitalization could potentially provide insight into whether the job posting is fradulent, most of the data takes the form of formal descriptions where emphasizing words with capitalization is unlikely. Next, all punctuation was replaced by whitespace, this is to further reduce noise since punctuation is not useful for our classification. Next, each string was converted to an array of strings where each element is a word, this will make language processing easier for the model. Finally, a list of shortwords were expunged from the data, this list includes common and often meaningless words for the purpose classification such as "and", "the", and "we".

### It is also worth noting a bug with the dataset caused by the storage format. New line characters are not captured in our data, meaning if the original description used new lines for punctuation, such as in a bullet point list, the last word of the previous line and the first word of the next line combine into a single word (for example documentsDesign). This could be mitigated by adding whitespace whenever a lowercase letter is immediately followed by an uppercase letter, but that could cause further problems where camel case is used intentionally, such as in company names.

### We tokenized our dataset which help make language preprocessing easier when using a neural network model. Tokenizing helps break larger phrases into smaller chunk for better analysis for sequence or patterns of words.

### Word2Vec + LSTM
- the another way to convert the string is using Word2Vec, which will convert each word to a vector. The words has similar meaning will near to each other in the space, and the vector's direction will show the relationship of words
- because we are going to add embedding layer in the model, so we need to create a convert matrix which will use in embedding layer
- to create the convert matrix, we need to first using tokenizer to convert each word to a number (ex. and -> 1, the -> 2, to -> 3.etc)
- and the `get_weight_matrix` function will create a matrix which map 1->the vector which  word "and" being convert by word2vec mode


# Methodology
### In developing the model we used the tokenized data to create values for each word. This allows each word in the neural network to have a weight and value when being passed into the model. We first tokenize the description columns so each word within the column would have a value associated with the word. For example, the word the would have a value of 1 so any the in the column would be replaced with 1. We then pad each row of the column with a certain length of 500 to ensure each row has the same length when passed as inputs to the neural network. We would then split the split the description data into training and testing data where 20 percent of the data would be allocated to testing. The last 80 would be used for training the model. We would do the same preprocessing method for company profile column. We would tokenize and assign values to each unique word while padding the length of each row size to be 500. The data would similarly be split to the description data in a 80 to 20 ratio.

### The next step was building a model. We started the first model to be relatively small and simple. The model would have 3 layers where the first layer was a tanh activation function with 500 nodes. The second layer would be a relu activation function with 100 nodes. We would then have the third layer be a sigmoid layer where the values would be converted between 0 and 1. 

### The model would then be fitted using the description data and the company profile data. The model would then predict for the two different tokenized data. The outputted result would be converted to binary values with a threshold of 0.5 because the y_test consists of binary values. We then printed the accuracy and classification report for description and company profile data. We would see that the description yielded a 95.7% accuracy while the company profile yielded a 93.5%.

# Data Visualization
### We created a word cloud for data visualization to see the most used words in the data. This allows us to see which words are more important and affects the outcome the most in the data set. For example, we can see that the most important words are team, will and work. These words appear most often in the descriptions for job offering. The model will use this information later to see if these words appear in real or fake job postings more often and can be used to identify them.

# Conclusions
