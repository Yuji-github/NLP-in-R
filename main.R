# Natural Language Processing 

# import data 
dataset_original = read.csv('Restaurant_Reviews.tsv', sep='\t', quote = '', stringsAsFactors = FALSE)
# stringsAsFactors = FALSE -> review does not recognize as a factor


# set libraries for cleaning the texts

# install.packages('tm') # for corpus
# install.packages('SnowballC') # for stopwards
library(tm)
library(SnowballC)


# cleaning the texts: get rid of unnecessary words from each rows
# step 1: turn into lower cases 

corpus = VCorpus(VectorSource(dataset_original$Review)) # for cleaning text in corpus
corpus = tm_map(corpus, content_transformer(tolower)) #turn into lower cases 
# print(as.character(corpus[[1]])) # checking the sentence of lower cases 


# step 2: removing all number from reviews 
corpus = tm_map(corpus, removeNumbers) 
# print(as.character(corpus[[841]])) # checking no numbers in a sentence


# step 3: removing any punctuation
corpus = tm_map(corpus, removePunctuation)
# print(as.character(corpus[[1]])) # checking no punctuation in a sentence


# step 4: remove non relevant words like 'this', 'a', 'the'
corpus = tm_map(corpus, removeWords, stopwords())
# print(as.character(corpus[[2]])) # missing 'not'


# step 5: stemming texts 
corpus = tm_map(corpus, stemDocument)
# print(as.character(corpus[[1]])) # checking loved -> love


# step 6: remove extra spaces
corpus = tm_map(corpus, stripWhitespace)
# print(as.character(corpus[[841]])) # check no extra space 


# creating the bag of words 
dtm = DocumentTermMatrix(corpus) # this will be our spase metrix

# ncol : in t1577 -> too much -> need filter
# print(dtm) 
# Sparsity           : 100% is a lot of zeros and no filter 

dtm = removeSparseTerms(dtm, 0.99) # 99% of words most frequency is not counted = count each ones 

# ncol : int 96
# print(dtm) # Sparsity           : 98%

dataset = as.data.frame(as.matrix(dtm)) # for transform space matrix for training 
dataset$Liked = dataset_original$Liked


# training data set (Random Forest/Naive Bays/Decision Tree) most common
library(caTools)
library(randomForest)

dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set

set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')

set.seed(123)
classifier = randomForest(x = training_set[-97],
                          y = training_set$Liked,
                          ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-97])

# Making the Confusion Matrix
cm = table(test_set[, 97], y_pred)

print(cm)