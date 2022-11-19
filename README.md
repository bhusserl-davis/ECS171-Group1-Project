# ECS 171 Group 1 Project

Groupmembers: Jade Elkins, Braxton Husserl, Victor Lai, Minh Giang Tran, Chuan Hsin Wang, Martin Wong

# Abstract
### Being able to differentiate between real and fake job postings would decrease the risk of applying for scam positions, leaving more time to apply for real, potential job offerings. We are planning on using a natural language processing scheme such as a recurrent neural network or transformer to predict whether the job listing is fraudulent or not. This dataset should give us a greater understanding of how to build models that work on natural language data.

### Dataset: https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs

### Description: The dataset contains 18k job descriptions, 800 of which are fake. It contains eighteen different features including job title, job description, salary, industry, and whether they are fraudulent or not.

# Background
# Introduction
## Data Exploration
### We first imported the data and printed the first 5 rows of the data in order to check if we imported the correct dataset. We then check for NAN/Null values within our dataset. We see that 12 of the 18 columns contain at least 1 NaN value within it. We decided to replace the NaN values with empty strings over dropping the rows and columns with NaN. We didn’t drop any of the rows with NaN because we would be dropping at least over 15,000 of the 17,880 observations we would have. Dropping columns of our dataset would result in a shrinkage of our dataset of over 83%. Replacing the NaN values with empty strings would allow us to still utilize the columns and observations without needing to remove a majority of the data.

### We then check numerical columns for unique data to see the ranges for each column and understand which columns had scaled or unscaled data. We see that telecommuting, company logo, questions, and fraudulent (the target class) have only 0 and 1 as unique values, implying they are boolean data. We can definitely use a sigmoid activation layer for our model to determine if the jobs are real or fakes.

### We created a word cloud for data visualization to see the most used words in the data. This allows us to see which words are more important and affects the outcome the most in the data set.

## Data Preprocessing
### All strings were converted to lowercase, that way strings like "Team" and "team" are not viewed differently by the model. While capitalization could potentially provide insight into whether the job posting is fradulent, most of the data takes the form of formal descriptions where emphasizing words with capitalization is unlikely. Next, all punctuation was replaced by whitespace, this is to further reduce noise since punctuation is not useful for our classification. Next, each string was converted to an array of strings where each element is a word, this will make language processing easier for the model. Finally, a list of shortwords were expunged from the data, this list includes common and often meaningless words for the purpose classification such as "and", "the", and "we".

### It is also worth noting a bug with the dataset caused by the storage format. New line characters are not captured in our data, meaning if the original description used new lines for punctuation, such as in a bullet point list, the last word of the previous line and the first word of the next line combine into a single word (for example documentsDesign). This could be mitigated by adding whitespace whenever a lowercase letter is immediately followed by an uppercase letter, but that could cause further problems where camel case is used intentionally, such as in company names.

# Methodology
# Data Visualization
# Conclusions
