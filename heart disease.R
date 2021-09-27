# Objective

# This dataset dates from 1988 and include some several databases of different country. On here I use this dataset for predict the statement of heart disease using Machine Learning Algorithms. 
# The main object is to predict whether the given patient is having heart disease or not. So it helped for that some several variables such as, age, sex, cholesterol level, blood pressure, types of chest pains etc.
# Here I used Machine Learning algorithms which are Decision Tree & Random Forest.

# Let’s begin by loading the required, needed libraries and importing the Heart_Disease.csv datasheet.
library(readr)
library(ggplot2)
library(Hmisc)
library(forcats)
library(dplyr)
library(psych)
library(C50)
library(caret)
library(randomForest)

df <- read_csv("newdataset/Heart_Disease.csv")

head(df)
# here head function gives us the top 6 examples of this data for much understanding.
# The independent variables are related to the patients details and based on those variables we select the dependent variable as statement of disease which have or not.

#age - age in years
#sex - sex of the patients 
#cp - chest pain type
#trestbps - resting blood pressure (in mm Hg on admission to the hospital)
#chol - serum cholesterols in mh/dl
#fbs - (fasting blood sugar > 120 mg/dl)
#restecg - resting electrocardiographic result (0- normal; 1- having ST-T wave abnormality; 2- showing probable)
#thalach - maximum heart rate achieved
#exang - exercise include angina 
#oldpeak - ST depression induced by exercise relative to rest
#slope - the slope of the peak exercise ST segment
#ca - number of major vessels (0-3) colored by fluoroscopy
#thal - thalassemia
#target - disease status

# According to nature of the features we implementing the changes
df$sex <- as.factor(df$sex)
df$cp <- as.factor(df$cp)
df$fbs <- as.factor(df$fbs)
df$restecg <- as.factor(df$restecg)
df$exang <- as.factor(df$exang)
df$slope <- as.factor(df$slope)
df$thal <- as.factor(df$thal)
df$target <- as.factor(df$target)

str(df)
# From the srt function we can get an idea about structure of the dataset to fine which are the characters and numeric.

Hmisc::describe(df)
# From describe function we can explore the content of the variable more than summary function. 
# According to these things we can extract some of the features of the data as graphically. Then we can observe the things more than just appearing.

# Correlations
pairs.panels(df[c("age", "trestbps", "chol", "thalach", "oldpeak")])
# From above output the diagonal, the scatterplots has been replaced with a correlation matrix. On the diagonal, a histogram depicting the distribution of values for each feature is shown. Finally, the scatterplots below the diagonal now are presented with additional visual information.
# So we have highest correlation between age & maximum heart. That is a negative (-0.4); lowest correlation between maximum heart & serum cholesterols (-0.01)
# (The oval-shaped object on each scatterplot is a correlation ellipse. It provides a visualization of how strongly correlated the variables are. The dot at the center of the ellipse indicates the point of the mean value for the x axis variable and y axis variable. The correlation between the two variables is indicated by the shape of the ellipse; the more it is stretched, the stronger the correlation.)

# Visualization of the gender
Hmisc::describe(df$sex)
ggplot(data = df) + 
  geom_bar(mapping = aes(sex), fill = c("lightyellow3", "lavenderblush3")) +
  labs(
    title = paste("Gender"),
    y = "Number of patients", 
    x = "Gender" ) +
  theme_dark()
# There are much more male patients than female. It’s like half of the male patients.

# Visualization of the chest pain type
Hmisc::describe(df$cp)
ggplot(data = df, 
       mapping = aes(cp))+
  geom_bar(fill = c(rep(c("aquamarine4", "aquamarine3",
                          "aquamarine2", "aquamarine1"),
                        times = 2))) +
  facet_wrap(~sex, nrow = 2) +
  scale_y_continuous(breaks = seq(0, 100, by = 10))+
  labs(
    title = "Chest pain types by Gender",
    x = "Pain Type", y = "Number of patients") +
  coord_flip() +
  theme_minimal()
# Here the typical angina is the most pain type of the chest of both sex. In addition to that there are male patients over 100 of that pain type.

# Visualization of the resting blood pressure Vs serum cholesterols (highest values)
Hmisc::describe(df[4:5])
(bp <- filter(df, trestbps >= 135, chol >= 250) %>%
  select(sex, chol, trestbps,fbs) %>%
    arrange(trestbps))

ggplot(bp, mapping = aes(x = chol, y =trestbps)) + 
  geom_point(col = "chartreuse")+ 
  facet_wrap(fbs~sex) +
  labs(
    title = "Resting blood pressure Vs Serum cholesterols",
    subtitle = "In order to Gender & Fasting blood sugar",
    caption = "-Highest values (trestbps >= 135, chol >= 250)",
    x = "Serum cholesterols", y = "Resting blood pressure"
  ) +
  scale_y_continuous(breaks = seq(140, 200, by = 10)) +
  scale_x_continuous(breaks = seq(250, 500, by = 25)) +
  theme_dark()
# fbs(fasting blood sugar > 120 mg/dl) is divided as True & False logistic variable with Gender. Exploring these graphs we can observe that only 4 male patients’ classify as high fbs and most of the patients are lower fbs who are male.

# Visualization of heart rate Vs age
Hmisc::describe(df[7:9])
ggplot(data = df, mapping = aes(x = age, y = thalach)) +
  geom_boxplot(fill = "grey25", col = "white") +
  geom_point(color = "darkgoldenrod1") +
  facet_wrap(restecg~exang~sex, nrow = 1) +
  scale_y_continuous(breaks = seq(80,200, by = 10)) +
  labs(
    title = "Maximum heart rate Vs Age",
    subtitle = "-In order to Electrocardiographic result, angina & Gender",
    x = "Age in year", y = "Maximum heart rate") +
  theme_dark()
# Maximum heart rate achieved has been taken here for explore the variation with age in years. Boxplot and scatter plot have carried out here by them we can see how the data point are varying with in order to Electrocardiographic result, angina & Gender.
# The resting electrocardiographic result are taken as; 0- normal, 1- having ST-T wave abnormality & 2- showing probable. And we can observe here only few patients have that problem.
# The exercise include angina are taken as Yes or No category
# In order to these things, we are able to explore the data as much easier.

#Visualization of thalassemia
Hmisc::describe(df[12:13])
ggplot(df, mapping = aes(x = ca)) +
  geom_bar(fill = "seagreen1") +
  facet_wrap(~thal ) +
  labs(
    title = "Thalassemia disease case variation",
    subtitle = "-In order to vessels colored by fluoroscop",
    x = "vessels colored by fluoroscop", y = "Number of patients") +
  theme_dark() 
# We can observed most of patients include in fixed defect thalassemia disease and few patients have normal case of thalassemia


# Performing Decision Tree Machine Learning Algorithm

# Preparation the data

# Firstly, we are dividing our dataset into training and test data as 80% and 20% of the initial. (243 records for training & 60 records for test)
# So creating that datasets we should consider about the random order of the data. 
# We’ll solve this problem randomly ordering our heart disease data set prior splitting. 
# The order() function is used to rearrange the list. For random number generation, we'll use runif() function.

set.seed(12345)
df_rand <- df[order(runif(303)), ]
df_train <- df_rand[1:243, ]
df_test <- df_rand[244:303, ]

# Just look what we have prepared, 
prop.table(table(df_train$target))
prop.table(table(df_test$target))
# This appears to be a fairly equal split, so we can now build our decision tree.

# Training a model on the data

# We will use the C5.0 algorithm for training our decision tree model.
heart_model <- C5.0(df_train[-14], df_train$target)

# We can see some basic data about the tree by just typing its name,
heart_model

# The tree is listed size of 22, which indicates that the tree is 22 decisions deep.
# To the decisions just type the summary() function.
summary(heart_model)
# The Errors field notes that the model correctly classified all, but 20 out of 243 training instances for an error rate of 8.2%.

# Improving model performance (Boosting the accuracy of decision trees)

# The C5.0() function makes it easy to add boosting to our C5.0 decision tree. We simply need to add an additional trials parameter indicating the number of separate decision trees to use in the boosted team.
heart_model <- C5.0(df_train[-14], df_train$target, trials = 11)
# By boosting we can perform the model as 0.4% model error given (we can simply see that just typing summary() function. So we can improve our model performance by reducing error rate 8.2% to 0.4%.

# Evaluating model performance
heart_predict <- predict(heart_model, df_test)
confusionMatrix(heart_predict, df_test$target)
# In this case we can get; Model Accuracy = 86.67%, Kappa = 73.36%, Sensitivity = 89.66%, Specificity = 83.87%. 

# Plotting

# In the final tree we have 11 separation decisions trees, by plot() function with trial parameter we can just plot each tree.
# Just exampling, it's has been plotted 4, 6 & 9 trials
plot(heart_model, trial = 4)
plot(heart_model, trial = 6)
plot(heart_model, trial = 9)


# Performing Random Forest Machine Learning Algorithm

# We'd like to test wether that our model performance is more by trying using Random Forest Algorithm
set.seed(300)
heart_rf <- randomForest(target ~ . , data = df)
heart_rf
# To look at a summary of the model's performance, we can simply type the resulting object's name, as expected, the output notes that the random forest included 500 trees and tried 3 variables at each split.
# It's seemingly poor re-substitution error according to the display confusion matrix, the error (OOB) rate of 16.5%.

# Evaluating random forest performance

#For the most accurate comparison of model performance, we'll use repeated 10-fold cross-validation: 10 times 10-fold CV.
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
grid_rf <- expand.grid(.mtry = c(2, 3, 4, 5))
set.seed(300)
(rf_model_heart <- train(target ~ . , data = df, method = "rf",
                metric = "Kappa", trControl = ctrl,
                tuneGrid = grid_rf))
plot(rf_model_heart)
# According to that result we'd better to take mtry as 2. Then we can predict more accuracy rate from our random forest model.

(heart_rf1 <- randomForest(df_train[-14], df_train$target,
                            ntree = 500, mtry = 2))

plot(varImp(rf_model_heart), main = "Variable Impotance for model")

plot(heart_rf1, main = "Trees Vs Class Error")
# From the graph the red line represent not having heart diseases and green represent having the diseases.
# The black line represent overall OOB error rate.

# Conclusions
# After performing two algorithm which are Decision Tree and Random Forest we can conclude both had good accuracy rate and out of which decision Tree gave a better accuracy of 86.7%. We can also say that it was misclassification 13.3%. 






