# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrgram)
library(caret) 
library(e1071)
library(pROC)
library(MASS) 
library(tidyr)

# Load the datasets
train_data <- read.csv("C:/Users/baral/OneDrive/Documents/train.csv")
test_data <- read.csv("C:/Users/baral/OneDrive/Documents/test_final.csv")

# View the structure of the dataset
str(train_data)
head(train_data)

# Check for missing values
missing_values <- colSums(is.na(train_data))
missing_values

# Summary statistics
summary(train_data)

# Data Preparation: Convert categorical variables to factors
train_data$Employer <- as.factor(train_data$Employer)
train_data$Education <- as.factor(train_data$Education)
train_data$Marital_Status <- as.factor(train_data$Marital_Status)
train_data$Occupation <- as.factor(train_data$Occupation)
train_data$Inc <- as.factor(train_data$Inc)

# Convert categorical variables in test_data to factors
test_data$Employer <- as.factor(test_data$Employer)
test_data$Education <- as.factor(test_data$Education)
test_data$Marital_Status <- as.factor(test_data$Marital_Status)
test_data$Occupation <- as.factor(test_data$Occupation)


str(train_data)

##############################
numeric_vars <- sapply(train_data, is.numeric)
par(mfrow = c(2, 2))
lapply(names(train_data)[numeric_vars], function(var) {
  boxplot(train_data[[var]], main = var, col = "lightblue")
})

cor_matrix <- cor(train_data[, numeric_vars], use = "complete.obs")

# Visualize correlations with a heatmap
cor_heatmap <- reshape2::melt(cor_matrix)
ggplot(cor_heatmap, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap", x = "", y = "")

#####################################

train_data %>%
  # Use dplyr::select to ensure using the correct function
  dplyr::select(Age, Capital, Hours) %>%
  # Use pivot_longer instead of gather
  pivot_longer(cols = c(Age, Capital, Hours), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  facet_wrap(~variable, scales = "free") +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  theme_minimal()


##############################################

# Step 3: Model Building
# Split the training data into training and validation sets
set.seed(123)
train_index <- createDataPartition(train_data$Inc, p = 0.8, list = FALSE)
train_set <- train_data[train_index, ]
valid_set <- train_data[-train_index, ]


################################

model_baseline <- glm(Inc ~ ., data = train_set, family = binomial)

# Predictions on validation set
pred_baseline <- predict(model_baseline, valid_set, type = "response")
pred_baseline_class <- ifelse(pred_baseline > 0.5, "High", "Low")


# Convert the predicted values to factors
pred_baseline_class <- factor(pred_baseline_class, levels = c("Low", "High"))
valid_set$Inc <- factor(valid_set$Inc, levels = c("Low", "High"))

# Calculate precision using posPredValue
precision_baseline <- posPredValue(pred_baseline_class, valid_set$Inc, positive = "High")

# Evaluation metrics
accuracy_baseline <- mean(pred_baseline_class == valid_set$Inc)
precision_baseline <- posPredValue(pred_baseline_class, valid_set$Inc, positive = "High")

print(accuracy_baseline)
print(precision_baseline)


###############################################



# Model 2: Logistic Regression with Backward Elimination

model_backward <- step(model_baseline, direction = "backward")

# Predictions on validation set
pred_backward <- predict(model_backward, valid_set, type = "response")
pred_backward_class <- ifelse(pred_backward > 0.5, "High", "Low")
pred_backward_class <- factor(pred_backward_class, levels = c("Low", "High"))

# Evaluation metrics
accuracy_backward <- mean(pred_backward_class == valid_set$Inc)
precision_backward <- posPredValue(pred_backward_class, valid_set$Inc, positive = "High")

print(accuracy_backward)
print(precision_backward)



##############################################

# Model 3: Logistic Regression with Stepwise Selection

model_stepwise <- step(glm(Inc ~ 1, data = train_data, family = binomial), 
                       scope = list(lower = ~1, upper = ~.), 
                       direction = "both")

# Predictions on validation set
pred_stepwise <- predict(model_stepwise, valid_set, type = "response")

pred_stepwise_class <- ifelse(pred_stepwise > 0.5, "High", "Low")
pred_stepwise_class <- factor(pred_stepwise_class, levels = c("Low", "High"))

# Evaluation metrics
accuracy_stepwise <- mean(pred_stepwise_class == valid_set$Inc)
precision_stepwise <- posPredValue(pred_stepwise_class, valid_set$Inc, positive = "High")

print(accuracy_stepwise)
print(precision_stepwise)



#############################################



# Model 4: Naive Bayes

model_nb <- naiveBayes(Inc ~ ., data = train_data)

# Predictions on validation set
pred_nb <- predict(model_nb, valid_set)
pred_nb <- factor(pred_nb, levels = c("Low", "High"))

# Evaluation metrics
accuracy_nb <- mean(pred_nb == valid_set$Inc)
precision_nb <- posPredValue(pred_nb, valid_set$Inc, positive = "High")

print(accuracy_nb)
print(precision_nb)


###########################################

# Model 5: LDA

# LDA model
model_lda <- lda(Inc ~ ., data = train_data)

# Predictions on validation set
pred_lda <- predict(model_lda, valid_set)$class
pred_lda <- factor(pred_lda, levels = c("Low", "High"))

# Evaluation metrics
accuracy_lda <- mean(pred_lda == valid_set$Inc)
precision_lda <- posPredValue(pred_lda, valid_set$Inc, positive = "High")

print(accuracy_lda)
print(precision_lda)


##############################################33
#4.
# Apply the LDA model to the test data
test_predictions <- predict(model_lda, test_data)

# Add predictions to the test dataframe
test_data$Predicted <- test_predictions$class
head(test_data)


########################################
#5

# Write the test data with predictions to a new file
write.csv(test_data, 'test_predictions.csv', row.names = FALSE)




