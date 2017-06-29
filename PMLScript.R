###################################################################################################
##
## Practical Machine Learning
## Adam Scarth
## June 29, 2017
##
## This program is designed to build a model on the Human Activity Recognition data from this study
## http://groupware.les.inf.puc-rio.br/har
##
## The goal is to be able to predict how well an activity is being done
##
## Instructions for setting up multicore processing can be found here
## https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md
##
###################################################################################################


#########################################
## Install packages and download files ##
#########################################


# Load packages
install.packages("caret")
install.packages("dplyr")
install.packages("parallel")
install.packages("doParallel")
library(caret)
library(dplyr)
library(parallel)
library(doParallel)


# Download files
fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileURL, "./pml-training.csv", method = "curl")
fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileURL, "./pml-testing.csv", method = "curl")


# Load data
trainingorig <- read.csv("./pml-training.csv")
testing20 <- read.csv("./pml-testing.csv")


###################################
## Data Exploration and Cleaning ##
###################################


# High level check of what the data is
summary(trainingorig)
head(trainingorig)
str(trainingorig)
dim(trainingorig)
names(trainingorig)
trainingorig[1:10, 160]
head(testing20)
str(testing20)
dim(testing20)
names(trainingorig)[1:20]
trainingorig$cvtd_timestamp
trainingorig$X
trainingorig$new_window


# Split to model training and test sets
set.seed(42)
inTrain <- createDataPartition(y = trainingorig$classe, p = 0.6, list = FALSE)
training <- trainingorig[inTrain, ]
testing <- trainingorig[-inTrain, ]


# Find features with too many NA's and remove
# An arbitrary threshold of 20% was used, but when values were missing in a column it was generally over 90% empty
naList <- colMeans(is.na(training))
qplot(naList)
colFilter <- naList < 0.2
names(training)[colFilter]
names(training)[!colFilter]
training <- training[, colFilter]
dim(training)


# Remove features 1 through 7 which contain experiment information vs. predictive values
names(training)[-(1:7)]
training <- training[, -(1:7)]
dim(training)
colMeans(is.na(training))
summary(training)


# Check for "" and "#DIV/0!" and remove
# When all "" were removed there were no more "#DIV/0!" remaining
str(training)
colSums(training == "")
colMeans(training == "")
colMeans(training == "") > 0.2
blankList <- colMeans(training == "") > 0.2
names(training)[blankList]
names(training)[!blankList]
training <- training[, !blankList]
dim(training)
colSums(training == "#DIV/0!")


# Explore features
featurePlot(x = training[,1:26], y = training$classe, par.strip.text = list(cex = 0.65))
featurePlot(x = training[,27:52], y = training$classe, par.strip.text = list(cex = 0.65))


# Principal component analysis
preProc <- preProcess(training[, -53], method = "pca", pcaComp = 2)
trainPC <- predict(preProc, training[, -53])
trainPC$classe <- training$classe
qplot(trainPC[, 1], trainPC[, 2], colour = training$classe)


# Review features
trainPC2 <- prcomp(training[, -53])
plot(trainPC2, type = "l", main = "Variance Reduction Graph", sub = "Number of Features")


###################
## Model Fitting ##
###################


# First, set up multicore processing, leaving one processor for the OS
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)


# Set up 10-fold cross validation and activite parallel processing
fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)


# Fit the random forest and review the results
modFit <- train(classe ~ ., data = training, method = "rf", trControl = fitControl)
modFit


# Deactivate the multicore processing
stopCluster(cluster)
registerDoSEQ()


# Review model fit on testing data
plot(varImp(modFit), scales = list(cex = 0.6))
pred <- predict(modFit, newdata = testing)
confusionMatrix(pred, testing$classe)


# Fit a CART model
fitControl <- trainControl(method = "cv", number = 10)
modFitTree <- train(classe ~ ., data = training, method = "rpart", trControl = fitControl)
predTree <- predict(modFitTree, newdata = testing)
confusionMatrix(predTree, testing$classe)


# Fit a gradient boosting model
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
modFitGBM <- train(classe ~ ., data = training, method = "gbm", trControl = fitControl, verbose = FALSE)
stopCluster(cluster)
registerDoSEQ()
predGBM <- predict(modFitGBM, newdata = testing)
confusionMatrix(predGBM, testing$classe)


# Final predicted results on the test data
pred20 <- predict(modFit, newdata = testing20)
finalresult <- data.frame(testing20$problem_id, pred20)
names(finalresult) <- c("problem_id", "rf_predicted")
finalresult








