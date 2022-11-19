# ECS 171 Group 1 Project

Groupmembers: Jade Elkins, Braxton Husserl, Victor Lai, Minh Giang Tran, Chuan Hsin Wang, Martin Wong

# Abstract
### Being able to differentiate between real and fake job postings would decrease the risk of applying for scam positions, leaving more time to apply for real, potential job offerings. We are planning on using a natural language processing scheme such as a recurrent neural network or transformer to predict whether the job listing is fraudulent or not. This dataset should give us a greater understanding of how to build models that work on natural language data.

### Dataset: https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs

### Description: The dataset contains 18k job descriptions, 800 of which are fake. It contains eighteen different features including job title, job description, salary, industry, and whether they are fraudulent or not.

# Background
# Introduction
# Data Exploration
### We first imported the data and printed the first 5 rows of the data in order to check if we imported the correct dataset. We then check for NAN/Null values within our dataset. We see that 12 of the 18 columns contain at least 1 NaN value within it. We decided to replace the NaN values with empty strings over dropping the rows and columns with NaN. We didn’t drop any of the rows with NaN because we would be dropping at least over 10,000 of the 17,880 observations we would have. Dropping columns of our dataset would result in a shrinkage of our dataset of over 50%. Replacing the NaN values with empty strings would allow us to still utilize the columns and observations without needing to remove over half the data.
#
### We then check numerical columns for unique data to see the ranges for each column and understand which columns had scaled or unscaled data. We see that telecommuting, company logo, questions, and fraud are within the range of 0 and 1. We can perceive that the result column, which is the fraudulent column, is either 0 for the real job and 1 for the real job. We can definitely use a sigmoid activation layer for our model to determine if the jobs are real or fakes.
### We created a word cloud for data visualization to see the most used words in the data. This allows us to see which words are more important and affects the outcome the most in the data set.
#
### The next steps taken are converting all words to lowercase and removing punctuations. This is important in removing noise from our dataset as it isn’t useful and can interfere with the text analysis if not fixed. We then tokenize the text data because it will make language processing easier for the model. Tokenize helps break larger phrases into smaller chunks for better analysis for sequence or patterns of words.

# Methodology
# Data Visualization
# Conclusions
