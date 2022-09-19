# 1) INTRODUCTION
## 1.1) Objective
#A naval mine is a self-contained explosive device
#placed in water to hinder, damage, or utterly 
#destroy naval ships. A Sonar (Sound Navigation 
#and Ranging) emits sound waves to locate and 
#avoid underwater hazards to navigation, but an
#experienced sonar operator is necessary to tune
#the equipment and analyze the submarine structures
#like rock or debris of similar size and shape.
#The presented algorithm raises the chance of 
#success of this operator at his job, giving 
#him a data-driven perspective and enhancing
#the likelihood of a better outcome for the
#assignment and survivability of the crew.

## 1.2) Metrics, Terminology, and Usability.
#The premise is that the maneuverability and 
#time cost of a stretched trip under enemy 
#waters surpasses the impact of losing the 
#ship or the crew. As the main reason for 
#the algorithm is to enhance the crew's 
#survivability, the metric to pursue is 
#Sensitivity to Mines. Sensitivity refers
#to the ability of the model to identify
#a mine correctly. It may seem evident at
#first glance, but every choice comes with
#a commitment relationship: It is reasonable
#for the model wrongly identify some debris
#as a mine if this ensures more mines are 
#correctly identified. Taking this into
#consideration, the purpose of the use
#is when advised by the sonar operator
#since employing it while cruising 
#through peaceful waters can drag the
#trip without gain. 
#For the sake of simplicity, debris on these
#documents will be called Rocks and 
#identified by R, while Mines will be 
#identified as M. 
#As with many sensible datasets, the 
#features already passed the security
#by obscurity procedure, with features
#with no understandable names and already
#standardized between 0 and 1 to prevent
#information leaks. This lack of meaning 
#can create an additional challenge.

# 2) ANALYSIS
## 2.1) PREPARING THE DATA
### 2.1.1) Installing the packages
#We will need tools to achieve our goal. 
#They need to be installed and loaded to 
#be called when we need them. We verify
#if we already have them and install 
#them if they are not present.

#Installing packages
if(!require(dslabs)) install.packages("dslabs")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(purrr)) install.packages("purrr")
if(!require(MASS)) install.packages("MASS")
if(!require(gam)) install.packages("gam")
if(!require(randomForest)) install.packages("randomForest")
if(!require(dplyr)) install.packages("dplyr")
if(!require(ggthemes)) install.packages
if(!require(knitr)) install.packages("knitr")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(summarytools)) install.packages("summarytools")
if(!require(ggstatsplot)) install.packages("ggstatsplot")
if(!require(PMCMRplus)) install.packages("PMCMRplus")

#Loading libraries
library(dslabs)
library(tidyverse)
library(caret)
library(purrr)
library(matrixStats)
library(MASS)
library(gam)
library(randomForest)
library(dplyr)
library(ggthemes)
library(knitr)
library(kableExtra)
library(summarytools)
library(ggstatsplot)
library(PMCMRplus)

### 2.1.2) Downloading the dataset
#This code downloads the dataset from GitHub.
#After that, it separates by commas and renames
#each column to reflect the features(x00) and 
#the outcome(y). An annually updated version 
#of the dataset can be seen in the Reference
#section of this document.

#Creates a tempfile
dl <- tempfile()

#Download from Github to tempfile [1]
download.file("https://raw.githubusercontent.com/wmdalboni/HarvardX.Capstone.Self/main/sonardataset.csv", dl)

#Separate the file by comma, ignoring headers,
#and save into the dataset variable
dataset <- read.csv(dl, header=FALSE,sep=",")

#view(dataset) #Check!

#Only three digits to an easier visualization.
options(digits=3)

# Let us keep the original dataset intact. 
#We may need it and do not want to download
#it again.
nds <- dataset

#view(nds) #Check!

#Forcing data.frame format.
nds <- as.data.frame(nds)

#Sweep the dataset, changing column names.
for(i in 1:ncol(nds)){
  #Rename the dependent variable column to y.
  if(i == ncol(nds)){
    colnames(nds)[i] <- "y" 
    #The y needs to be a factor.
    nds[,i] <-  nds[,i] %>% as.factor()
  } else {
    #Rename the independent variables column to x01, x02, x03...
    colnames(nds)[i] <- paste("x",str_pad(i,2,pad ="0"),sep="")
    #The x needs to be numeric.
    nds[,i] <-  nds[,i] %>% as.numeric()
  }
}
#glimpse(nds) #Check!

## 2.2) EXPLORATION & INSIGHTS
#Observing the datasets is essential to 
#get insights into what we got and what
#we can create to predict better.

### 2.2.1) Heads
#The "head" command shows us the six 
#rows from the top of each set to find 
#if they have the same names, column order,
#and data types. We have 60 different 
#features as can be seen.

#Transpose to better visualization
head(nds)[1:10] 
head(nds)[11:20]
head(nds)[21:30]
head(nds)[31:40]
head(nds)[41:50]
head(nds)[51:60]
head(nds)[61]

### 2.2.2) Glimpse
#Glimpse helps us pay attention to the
#columns' dimensions and variable types.
#Even though the dataset is relatively 
#small on rows, it has a fair amount of 
#features and probably will be enough 
#for a good prediction.

glimpse(nds)

### 2.2.3) Summary
#The summary shows the Quartiles, the 
#Minimum, and Maximum values. It is 
#crucial since it can show outliers not
#apparent using other approaches. This 
#time it becomes evident that the data
#was previously normalized between 0 and 1.
#As we can observe, there is no missing data
#between the features, which will contribute
#to faster preprocessing.

#Save the transposed summary
sum_nds <- t(summarytools::descr(nds))
t(sum_nds[,1:7]) #Only the columns needed

## 2.3) GRAPHS
### 2.3.1) Boxplot: Mines(RED) x Rocks(BLUE)
#Sometimes boxplots can help to notice patterns
#we can explore to create a better prediction
#or even remove outliers. There is a good 
#amount of variance between each feature, and
#probably there is no near zero variance feature.
#While there are features where mines tend to 
#have a higher median, others are lower or very
#close to rocks. 

#Creates a boxplot with Mine & Rocks
nds %>%
  dplyr::select(-y) %>% #Remove Outcome
  boxplot(col="white", #White Boxes
          xlab="Mines & Rocks") #The x Label

#Creates a boxplot with Mines only
nds %>%
  filter(y=='M') %>% #Filter by Mines
  dplyr::select(-y) %>% #Remove Outcome
  boxplot(col="red", #Red boxes
          xlab="Only Mines") #The x Label

#Creates a boxplot with Rocks only
nds %>%
  filter(y=='R') %>% #Filter by Rocks
  dplyr::select(-y) %>% #Remove Outcome
  boxplot(col="blue",#Blue boxes
          xlab="Only Rocks") #The x Label

### 2.3.2) Correlation Matrix: 
#More colors mean more correlation. There 
#are features with a strong correlation, 
#which may mean they represent confounded
#data, or it simply does not add much more
#information to the final model. We can 
#choose only some of them to use and drop 
#the others to optimize the process.

forcormt <- cor(nds[1:60]) %>% #Only the predictors
  as.data.frame() %>% #Save as data frame
  rownames_to_column() %>% #Create a column with the name of the predictors' columns
pivot_longer(-rowname) %>%# Increase the number of rows and decreasing the number of columns to fit the plot
  arrange(desc(value)) #Order highest to lowest to an easier check of the dataset

#view(forcormt) #Check!

forcormt %>%
  ggplot(aes(x=rowname, y=name, fill=value)) +
  geom_tile()+ #Matrix graph
  scale_fill_gradient2(low = "red",#Scores under 0
                       high = "blue", #Scores above 0
                       mid= "white", # No correlation
                       na.value = "grey50", # NA
                       guide = "colourbar", # Color guide
                       aesthetics = "fill") + # Fill the tiles
  labs(
    title = "CORRELATION MATRIX", 
    subtitle = "More color means more correlation",
    x = "", # No label on X to gain more space
    y = "FEATURES x FEATURES",
    size = 8, 
    family = "sans") + 
  theme( # Details of the theme
    legend.position="none", #  No legend to gain more space
    title= element_text(colour = "black",
                        face="bold",
                        family="sans"), 
    axis.text.x = element_text(angle=90, #90 degrees
                               size = 8, 
                               face="bold",
                               colour = "black", 
                               family="sans", 
                               vjust=-0.05), # Adjust x label
    axis.text.y = element_blank()) #No label on y

# 3) PREPROCESSING: 
#DATA DROPPING & FEATURE SELECTION
## 3.1) Splitting the dataset
#There are organizational and functional 
#arguments to divide the dataset at this point.
#The dataset is separated to make the test set
#the real-life simulation to be predicted, and
#having its data leaked into our train set by
#some calculation could make the model less
#predictive than it could be. Even though some
#preprocessing topics under this could be done
#without separating the dataset, doing it here
#helps the cadence of reading and organization
#of headings.

#Set seed to be reproducible
set.seed(87, sample.kind = "Rounding")
#Generate dataset index by a percentage 
#of the dataset with a similar proportion 
#of y on each set
partition_index <- caret::createDataPartition(nds$y, 
                                              list=FALSE,
                                              p=0.2) #80x20
#partition_index #Check!

#xy = Features and Predictor
#x = Only features
#y = Only predictors
test_xy <- nds[partition_index,] 
train_xy <- nds[-partition_index,]

test_y <- nds[partition_index,] %>% dplyr::select(y)
train_y <- nds[-partition_index,] %>% dplyr::select(y)

test_x <- test_xy %>% dplyr::select(-y)
train_x <- train_xy %>% dplyr::select(-y)

#Check if they have similar proportions
#mean(train_y=="M") #Check!
#mean(test_y=="M") #Check!

## 3.2) Near Zero Variance
#It is not uncommon to have columns of 
#independent variables with near zero 
#variance between the values. If not 
#removed, they can negatively impact 
#the result of the final model. We did 
#not observe this possibility by glimpsing
#the dataset before, and now this code 
#proves it.

#zeroVar and nzv TRUE would need to be treated,
#but there is none.
nzv_x <- nearZeroVar(train_x, saveMetrics= TRUE)

nzv_x[1:20,]
nzv_x[21:40,]
nzv_x[41:60,] #Only the columns needed

## 3.3) High correlated features
#This dataset is relatively small but has
#many columns, and it is possible that some
#of them are so correlated and confounded 
#that it is possible to use only one of 
#them and achieve the same or even better
#results. The theory says 0.7 means a high
#correlation, so we will keep with it to make
#an arbitrary value of the cut. The correlation
#strength value can be tuned in the future if 
#the final results are unsatisfactory.

#Computes the correlation
cor_trx <- cor(train_x) #Computes the correlation
#glimpse(cor_trx) #Check!

#Finding the indexes of columns with a correlation of more than 0.7
high_cor_trx <- findCorrelation(cor_trx, cutoff =.7) 
#view(high_cor_trx) #Check!

#Arbitrary cut
train_x_uncor <- train_x[,-high_cor_trx]
test_x_uncor <- test_x[,-high_cor_trx]

#high_cor_trx #Columns to drop

## 3.4) Linear combinations
#Some methods we may want to use can give a 
#lot worst results if they do not have linear 
#independence. The code would create a list 
#with elements for each dependency containing 
#vectors of column numbers as an index to be 
#removed if needed posteriorly. 

#Find if there are any linear combinations between the features.
#If the main dataset presents no linear combos, the non 
#correlated should follow the same rule.
combo <- findLinearCombos(train_x_uncor)
#combo$remove #Check! It is NULL, so there is nothing to remove.

#As can be seen, there is no such case.

## 3.5) Center & Scaling
#The process of centering and scaling the
#features enables the code to handle highly 
#varying magnitudes between columns. If not 
#done, the algorithms will fail to weigh the
#features since the most significant values 
#will be seen as more critical, regardless of
#the magnitude unit of the values.
#Each column's mean is calculated and divided
#by its own standard deviation. After this 
#technique, the values will be calculated as
#standard deviations from the mean and will 
#be mathematically comparable.
#The dataset was previously normalized between
#0 and 1 and could be used as is, but the 
#centered and scaled perspective can give 
#new insights.

#No Scientific notation 
options(scipen=999) 

#Force the same outcome every time.
set.seed(87, sample.kind = "Rounding")

preProcValues <- preProcess(train_x_uncor,
                            method = c("center", #Center
                                       #average as 0
                                       "scale")) #Scale the
                                       #difference around the
                                       #average.

#Using the preprocessed values around train
#and not the entire dataset may prevent data
#leak because another way the average would not
#be from the train set but the whole dataset.
transf_trx <- predict(preProcValues, train_x_uncor)
transf_tex <- predict(preProcValues, test_x_uncor) 

#transf_trx #Check!
#transf_tex #Check!

#Bind the outcome to the features we are
#going to use.
transf_trxy <- cbind(transf_trx,train_y)
transf_texy <- cbind(transf_tex,test_y)

#The boxplot here has the same purpose as 
#2.3.1) Boxplot: Mines(RED) x Rocks(BLUE) but
#uses only train set data. Some people may find
#it easier to understand, so it is here not only
#to analyze the train set but as a comparison 
#between normalization and the center and scale
#method.

#Creates a boxplot with the Train set
transf_trx %>%
    boxplot(col="white", #White Boxes
            xlab="Mines & Rocks",
            ylim=c(min(transf_trx),
                   max(transf_trx))) #Y axis size

#Creates a boxplot only with Mines
transf_trxy %>%
    filter(y=='M') %>% #Filter by Mines
    dplyr::select(-y) %>% #Remove Outcome
    boxplot(col="red", #Red Boxes
            xlab="Only Mines",
            ylim=c(min(transf_trx),
                   max(transf_trx))) #Y axis size

#Creates a boxplot only with Rocks
transf_trxy %>%
    filter(y=='R') %>% #Filter by Rocks
    dplyr::select(-y) %>% #Remove Outcome
    boxplot(col="blue", #Blue Boxes
            xlab="Only Rocks",
            ylim=c(min(transf_trx),
                   max(transf_trx))) #Y axis size

## 3.6) Outliers
#As can be seen, some high values could be
#considered outliers, but intentionally they
#will be untouched. This comes from the
#perspective that the data passed through a 
#process of security through obscurity; 
#in other words, the features have been 
#intentionally given no meaningful names and
#standardized from 0 to 1 to prevent too much
#knowledge of what the observation means and 
#which equipment was used to collect them. 
#Given this fact, there is a high chance of 
#biasing the data if the outliers' removal is
#done since the data may already have passed 
#previously by this process, and we lack the
#knowledge of each feature to make a sound 
#judgment. We will endure and work with what
#we get the best we can.

# 4) METHODOLOGY
## 4.1) Ten-fold Cross Validation
#The code instructs that the train_data must
#be folded into ten parts every time, with nine
#folds being used to calculate and one to 
#validate the results. Every fold will be used
#once to validate so that it will be done ten 
#times. There is always a chance of biasing 
#the data by that particularly arbitrary cut
#every time we split data at random, and more
#so knowing our dataset is reasonably few on
#rows, so the process will be repeated five
#times and take the mean to balance the 
#chance of random unlucky folds biasing
#the results.

fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, #10 Folds
                           repeats = 5) #5 Times

#Creating a data frame to hold the results
model_frame <- data.frame(matrix(ncol = 3, nrow = 0))
cnames <- c("Method", "Sensitivity", "Accuracy")
colnames(model_frame) <- cnames

## 4.2) Methods
#Multiple classification methods will be 
#used, each with their hyperparameters tuned
#when needed and, in the end, have their 
#Sensitivity to mine and overall Accuracy
#compared. Even though Sensitivity is the 
#critical value, it is essential to ensure 
#Accuracy is not too low. If the method always 
#says it is a mine, the Sensitivity would be 
#perfect, but the overall Accuracy would be low.
#Looking at both prevents this mistake.

### 4.2.1) Logistic Regression (GLM Binomial)

#The seed to be reproducible with the same results
set.seed(87, sample.kind = "Rounding")

#Train the model
train_glm <- train(x=transf_trx, #Features
                   y=transf_trxy$y,#Outcomes
                   method = "glm", #method
                   family= "binomial",#Logit link Binomial
                   trControl=fitControl) #Folds & Repeats
#Saves the prediction
glm_preds <- predict(train_glm, transf_tex) 

#Saves the results
glm_conf <- confusionMatrix(data=glm_preds,
                            reference= transf_texy$y)


#Saving the results to further comparison
model_frame <- rbind(model_frame,
                     c("glm",
                       glm_conf$byClass[[1]],
                       glm_conf$byClass[[11]]
                     )
)
#Reinforce the column names                     
colnames(model_frame) <- cnames

### 4.2.3) Linear Discriminant Analysis (LDA)

set.seed(87, sample.kind = "Rounding")

train_lda <- train(x= transf_trx,
                   y= transf_trxy$y,
                   method = "lda",
                   trControl=fitControl)

lda_preds <- predict(train_lda,  transf_tex)

lda_conf <- confusionMatrix(data=lda_preds,
                            reference= transf_texy$y)

model_frame <- rbind(model_frame,
                     c("lda",
                       lda_conf$byClass[[1]],
                       lda_conf$byClass[[11]]
                     )
)

colnames(model_frame) <- cnames

### 4.2.4) Quadratic Discriminant Analysis (QDA)

set.seed(87, sample.kind = "Rounding")

train_qda <- train(x= transf_trx,
                   y= transf_trxy$y,
                   method = "qda",
                   trControl=fitControl)

qda_preds <- predict(train_qda,  transf_tex)

qda_conf <- confusionMatrix(data=qda_preds,
                            reference= transf_texy$y)

model_frame <- rbind(model_frame,
                     c("qda",
                       qda_conf$byClass[[1]],
                       qda_conf$byClass[[11]]
                     )
)
colnames(model_frame) <- cnames

### 4.2.5) Local Polynomial Regression Fitting (gamLoess)

set.seed(87, sample.kind = "Rounding")

train_loess <- train(x=transf_trx,
                     y=transf_trxy$y,
                     method = "gamLoess",
                     trControl = fitControl)

loess_preds <- predict(train_loess, transf_tex)

loess_conf <- confusionMatrix(data=loess_preds,
                              reference=transf_texy$y)

model_frame <- rbind(model_frame,
                     c("loess",
                       loess_conf$byClass[[1]],
                       loess_conf$byClass[[11]]
                     )
)

colnames(model_frame) <- cnames

### 4.2.6) k Nearest Neighbors (kNN)

set.seed(87, sample.kind = "Rounding")

#The k nearest neighbors need a single number of 
#neighbors to be calculated. It will try multiple
#hyperparameters here to use the best one.
tuning_knn <- data.frame(k = seq(2,30,2)) 

train_knn <- train(x=transf_trx,
                   y=transf_trxy$y,
                   method = "knn", 
                   tuneGrid = tuning_knn, #As above.
                   trControl = fitControl)

knn_preds <- predict(train_knn, transf_tex)

knn_conf <- confusionMatrix(data=knn_preds,
                            reference=transf_texy$y)

model_frame <- rbind(model_frame,
                     c("knn",
                       knn_conf$byClass[[1]],
                       knn_conf$byClass[[11]]
                     )
)

colnames(model_frame) <- cnames

### 4.2.7) Random Forest (RF)

set.seed(87, sample.kind = "Rounding")

#Number of variables randomly sampled 
#as candidates at each split.
tuning_rf <- data.frame(mtry = seq(2,10,2))

train_rf <- train(x=transf_trx,
                  y=transf_trxy$y,
                  method = "rf",
                  tuneGrid = tuning_rf, #As above.
                  trControl= fitControl,
                  importance = TRUE)

#train_rf$bestTune #Check

rf_preds <- predict(train_rf, transf_tex)

rf_conf <- confusionMatrix(data=rf_preds,
                           reference=transf_texy$y)

model_frame <- rbind(model_frame,
                     c("rf",
                       rf_conf$byClass[[1]],
                       rf_conf$byClass[[11]]
                     )
)
colnames(model_frame) <- cnames

# 5) RESULTS

#order by Sensitivity, our critical value
ord_res_tb <- model_frame %>% 
  arrange(desc(Sensitivity))

#Saving to output when needed
out_res <- kable(ord_res_tb) %>%
  kable_paper()

#Forcing position of the table
kable_styling(out_res,
              latex_options = "hold_position")

# 6) CONCLUSION
#The Random Forest model can be used 
#on top of the sonar operator's expertise
#to enhance the operation's success without
#an overall accuracy that could hinder the
#ship's advance too much to be unusable. 
#There is more room for improvement with 
#more time and computational power invested
#into a more robust tunning. Besides that,
#the model could benefit from a newer version
#of the dataset with more observations, and
#knowing more about each feature could help
#narrow the outliers even more.

# 7) REFERENCES
## 7.1) SONAR MINE DATASET
#[1] DALVI, M. (2021, JULY). SONAR MINE DATASET,
#Version 1. Retrieved August 22, 2022, from
#https://www.kaggle.com/datasets/mayurdalvi/sonar-mine-dataset
