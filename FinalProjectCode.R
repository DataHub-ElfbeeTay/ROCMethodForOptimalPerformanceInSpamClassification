rm(list=ls())

#load necessary libraries
library(e1071)
library(caTools)
library(caret)
library(ROCR)
library(tidymodels)

# Read dataset
spam = read.table("spambase.data", header = F, sep = ",")
#set seed so the results of the split are reproducible
set.seed(1024)
# using the max-min normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
spamnormalizedf <- as.data.frame(lapply(spam, normalize))

#split dataset into training and test sets
spamnormalizedf$split <- sample.split(spamnormalizedf$V58, SplitRatio = 0.7 ) #70% training, 30% test
spam.training <- subset(spamnormalizedf, spamnormalizedf$split == TRUE)
spam.test <- subset(spamnormalizedf, spamnormalizedf$split == FALSE)
spam.training$split <- NULL
spam.test$split <- NULL
table(spam.test$V58)

# Check dimensions
dim(spam.training)
dim(spam.test)

# Check variable classes
sapply(spam.training, class)

sapply(spam.test, class)

# Summary statistics
summary(spam.training)
summary(spam.test)

# Check for NA columns
colSums(is.na(spam.training))[colSums(is.na(spam.training)) > 0]
colSums(is.na(spam.test))[colSums(is.na(spam.test)) > 0]

# Recode integers to numeric
spam.training[56:57] = lapply(spam.training[56:57], as.numeric)
spam.test[56:57] = lapply(spam.test[56:57], as.numeric)

# Recode integers to factor
spam.training$V58 = as.factor(spam.training$V58)
spam.test$V58 = as.factor(spam.test$V58)

# Set factor variable levels. Key: Not_Spam: 0, Spam: 1
levels(spam.training$V58) = c("Not_Spam", "Spam")
levels(spam.test$V58) = c("Not_Spam", "Spam")

#--------------------------------------
# Variable names as listed in spambase.names
#--------------------------------------
# Store names
spam.cn.all = c("word_freq_make",
                "word_freq_address",
                "word_freq_all",
                "word_freq_3d",
                "word_freq_our",
                "word_freq_over",
                "word_freq_remove",
                "word_freq_internet",
                "word_freq_order",
                "word_freq_mail",
                "word_freq_receive",
                "word_freq_will",
                "word_freq_people",
                "word_freq_report",
                "word_freq_addresses",
                "word_freq_free",
                "word_freq_business",
                "word_freq_email",
                "word_freq_you",
                "word_freq_credit",
                "word_freq_your",
                "word_freq_font",
                "word_freq_000",
                "word_freq_money",
                "word_freq_hp",
                "word_freq_hpl",
                "word_freq_george",
                "word_freq_650",
                "word_freq_lab",
                "word_freq_labs",
                "word_freq_telnet",
                "word_freq_857",
                "word_freq_data",
                "word_freq_415",
                "word_freq_85",
                "word_freq_technology",
                "word_freq_1999",
                "word_freq_parts",
                "word_freq_pm",
                "word_freq_direct",
                "word_freq_cs",
                "word_freq_meeting",
                "word_freq_original",
                "word_freq_project",
                "word_freq_re",
                "word_freq_edu",
                "word_freq_table",
                "word_freq_conference",
                "char_freq_semicolon",
                "char_freq_left_paren",
                "char_freq_left_bracket",
                "char_freq_exclamation",
                "char_freq_usd",
                "char_freq_pound",
                "capital_run_length_average",
                "capital_run_length_longest",
                "capital_run_length_total",
                "y")

# Replace placeholder column names on dataset with the above
colnames(spam.training) = spam.cn.all
colnames(spam.test) = spam.cn.all

#--------------------------------------
# Staging
#--------------------------------------
# Store dataset name for use in titles, etc. later
data.name <- "spam$"

# Set response variable
data.response <- "y"

# Assign numeric column names
spam.cn.num = colnames(spam.training[, -58])
m <- naiveBayes(y ~ ., data = spam.training)
p <- predict(m, spam.test[,-58])
table(p, spam.test$y)
df <- (confusionMatrix(spam.test[,58], p))
df
df$byClass
ctable <- as.table(matrix(c(135, 117, 701, 427), nrow = 2, byrow
                          = TRUE))
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"), conf.level =
               0, margin = 1, main = "Confusion Matrix",)

# plot histogram of predicted probabilities
# note overconfident predictions
probs <- predict(m, spam.test[,-58], type="raw")
naive_probs <- as.data.frame(probs)
result_naive <- data.frame(actual = spam.test$y, prediction = naive_probs$Spam)

#qplot(x=probs[, "Spam"], geom="histogram")
# plot ROC curve
pred_naive2 <- prediction(probs[, "Spam"], spam.test[, 58])
perf.rocr_naive = performance(pred_naive2, measure = "auc", x.measure =
                                "cutoff")
perf.tpr.rocr_naive = performance(pred_naive2, "tpr","fpr")
plot(perf.tpr.rocr_naive, col="black",main=paste("AUC:",(perf.rocr_naive@y.values)))

#To get their Roc
theme_set(theme_bw())
levels(result_naive$actual) = c("0", "1")
comb <- unique(result_naive$prediction)
comb <- as.vector(comb)
comb <- comb[-79:-100]
new0 <-c()
new1 <-c()
new2 <-c()
new3 <-c()
new4 <-c()
new5 <-c()
new6 <-c()
for(i in 1:length(comb)) {
  result_naive<- result_naive %>%
    mutate(pred_class = if_else(prediction>= comb[i], "1", "0")) %>%
    mutate_at(vars(actual, pred_class), list(. %>% factor() %>% forcats::fct_relevel("1")))
  #To calculate the sensitivity
  new0[i] <-result_naive %>%
    sens(actual, pred_class)%>%
    select(.estimate)
  new1[i] <- result_naive %>%
    spec(actual, pred_class)%>%
    select(.estimate)
  new3[i] <-result_naive %>%
    precision(actual, pred_class)%>%
    select(.estimate)
  new4[i] <-result_naive %>%
    recall(actual, pred_class) %>%
    select(.estimate)
  new5[i] <-result_naive %>%
    accuracy(actual, pred_class) %>%
    select(.estimate)
  new6[i] <-result_naive %>%
    f_meas(actual, pred_class) %>%
    select(.estimate)
}
comb <- as.data.frame(comb)
comb <- t(comb)
new0 <- as.data.frame(new0)
new0 <- t(new0)
new1 <- as.data.frame(new1)
new1 <- t(new1)
new3 <-as.data.frame(new3)
new3 <- t(new3)
new4 <-as.data.frame(new4)
new4 <- t(new4)
new5 <-as.data.frame(new5)
new5 <- t(new5)
new6 <-as.data.frame(new6)
new6 <- t(new6)
neural_compare11 <- cbind(new0,new1,new3,new4,new5,new6)
comb <- t(comb)
neural_compare11 <- cbind(comb,neural_compare11)
colnames(neural_compare11)<-
  list("threshold","sensitivity","specificity","precision","recall","accuracy","f-measure")
neural_compare11 <- as.data.frame(neural_compare11)
data_out <- write.csv(neural_compare11, "neural_compare3.csv",row.names = FALSE,col.names =
                        TRUE)
roc_dat <- result_naive %>%
  roc_curve(actual, prediction)
roc_dat
pr_dat <- result_naive %>%
  pr_curve(actual, prediction)
pr_dat
roc_dat %>%
  arrange(.threshold) %>% # this step is not strictly necessary here because the rows are already ordered by
  `.threshold`
ggplot() +
  geom_path(aes(1 - specificity, sensitivity)) + # connect the points in the order in which they appear in the data to form a curve
geom_abline(intercept = 0, slope = 1, linetype = "dotted") + # add a reference line by convention
  coord_equal()
pr_dat %>%
  arrange(.threshold) %>% # this step is not strictly necessary here because the rows are already ordered by
  `.threshold`
ggplot() +
  geom_path(aes(recall, precision)) + # connect the points in the order in which they appear in the data to form a curve
coord_equal()

#This gets the optimal threshold from the list of thresholds
#method 1
threshold1 <- function(predict, response) {
  perf <- ROCR::performance(ROCR::prediction(predict, response), "sens", "spec")
  df <- data.frame(cut = perf@alpha.values[[1]], sens = perf@x.values[[1]], spec = perf@y.values[[1]])
  df[which.max(df$sens + df$spec), "cut"]
}
#Method 2
threshold2 <- function(predict, response) {
  r <- pROC::roc(response, predict)
  r$thresholds[which.max(r$sensitivities + r$specificities)]
}
threshold1(result_naive$prediction, result_naive$actual)
threshold2(result_naive$prediction, result_naive$actual)

#Using the threshold gotten to get sensitivity,specificity et al
result_naive <- result_naive %>%
  mutate(pred_class = if_else(prediction>= 1, "1", "0")) %>%
  #mutate(pred_class = if_else(prediction>= 0.5, "1", "0")) %>%
  mutate_at(vars(actual, pred_class), list(. %>% factor() %>% forcats::fct_relevel("1")))

#To calculate the sensitivity
result_naive %>%
  sens(actual, pred_class)
result_naive %>%
  spec(actual, pred_class)
result_naive %>%
  spec(actual, pred_class)
result_naive %>%
  precision(actual, pred_class)
result_naive %>%
  recall(actual, pred_class)
result_naive %>%
  accuracy(actual, pred_class)
result_naive %>%
  f_meas(actual, pred_class)


#Confusion Matrix
mat <- confusionMatrix(result_naive$pred_class, result_naive$actual, positive="1")
#Confusion matrix
mat$table



#Working with neural network

# Clear working environment
rm(list=ls())

#load necessary libraries
library(e1071)
library(caTools)
library(caret)
library(ROCR)
library(tidymodels)
library(neuralnet)
library(dplyr)


# Read dataset
spam = read.table("spambase.data", header = F, sep = ",")

#set seed so the results of the split are reproducible
set.seed(1024)

# using the max-min normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
spamnormalizedf <- as.data.frame(lapply(spam, normalize))

#split dataset into training and test sets
spamnormalizedf$split <- sample.split(spamnormalizedf$V58, SplitRatio = 0.7 ) #70% training, 30% test
spam.training <- subset(spamnormalizedf, spamnormalizedf$split == TRUE)
spam.test <- subset(spamnormalizedf, spamnormalizedf$split == FALSE)
spam.training$split <- NULL
spam.test$split <- NULL
table(spam.test$V58)

# Check dimensions
dim(spam.training)
dim(spam.test)

# Check variable classes
sapply(spam.training, class)
sapply(spam.test, class)

# Summary statistics
summary(spam.training)
summary(spam.test)

# Check for NA columns
colSums(is.na(spam.training))[colSums(is.na(spam.training)) > 0]
colSums(is.na(spam.test))[colSums(is.na(spam.test)) > 0]

# Recode integers to numeric
spam.training[56:57] = lapply(spam.training[56:57], as.numeric)
spam.test[56:57] = lapply(spam.test[56:57], as.numeric)

# Note that Key: Not_Spam: 0, Spam: 1
#--------------------------------------
# Variable names as listed in spambase.names
#--------------------------------------
# Store names
spam.cn.all = c("word_freq_make",
                "word_freq_address",
                "word_freq_all",
                "word_freq_3d",
                "word_freq_our",
                "word_freq_over",
                "word_freq_remove",
                "word_freq_internet",
                "word_freq_order",
                "word_freq_mail",
                "word_freq_receive",
                "word_freq_will",
                "word_freq_people",
                "word_freq_report",
                "word_freq_addresses",
                "word_freq_free",
                "word_freq_business",
                "word_freq_email",
                "word_freq_you",
                "word_freq_credit",
                "word_freq_your",
                "word_freq_font",
                "word_freq_000",
                "word_freq_money",
                "word_freq_hp",
                "word_freq_hpl",
                "word_freq_george",
                "word_freq_650",
                "word_freq_lab",
                "word_freq_labs",
                "word_freq_telnet",
                "word_freq_857",
                "word_freq_data",
                "word_freq_415",
                "word_freq_85",
                "word_freq_technology",
                "word_freq_1999",
                "word_freq_parts",
                "word_freq_pm",
                "word_freq_direct",
                "word_freq_cs",
                "word_freq_meeting",
                "word_freq_original",
                "word_freq_project",
                "word_freq_re",
                "word_freq_edu",
                "word_freq_table",
                "word_freq_conference",
                "char_freq_semicolon",
                "char_freq_left_paren",
                "char_freq_left_bracket",
                "char_freq_exclamation",
                "char_freq_usd",
                "char_freq_pound",
                "capital_run_length_average",
                "capital_run_length_longest",
                "capital_run_length_total",
                "y")

# Replace placeholder column names on dataset with the above
colnames(spam.training) = spam.cn.all
colnames(spam.test) = spam.cn.all

neuural_model <- neuralnet(spam.training$y ~.,spam.training,hidden = c(2,1),linear.output = FALSE)
neuural_model$result.matrix
plot(neuural_model)
# 0.9326087 for c(2,1)

# Assign numeric column names
spam.cn.num = colnames(spam.training[, -58])

#The "subset" function is used to eliminate the dependent variable from
#the test data
spam_test <- subset(spam.test, select = spam.cn.num)
head(spam_test)

spam_result <- neuralnet::compute(neuural_model,spam_test)
result <- data.frame(actual = spam.test$y, prediction = spam_result$net.result)

#Computing the confusion
round_result<-sapply(result,round,digits=0)
round_result_df=data.frame(round_result)
attach(round_result_df)
predict.table <- table(actual,prediction)
confusionMatrix(predict.table)

#The model generates 789 true negatives (0's), 498 true positives (1's),
#while there are 46 false negatives and 47 false positives.
accuracy = (781+500)/(781+500+44+55)
detach(package:neuralnet,unload = T)

nn.pred = prediction(spam_result$net.result, spam.test$y)

#pref <- performance(nn.pred, "tpr", "fpr")
#auc <- as.numeric(performance(nn.pred, measure = "auc")@y.values)
#plot(pref,lwd=2, col="blue", main="ROC Curve for neural network")
#abline(a=0, b=1)

perf.rocr = performance(nn.pred, measure = "auc", x.measure =
                          "cutoff")
perf.tpr.rocr = performance(nn.pred, "tpr","fpr")
plot(perf.tpr.rocr, col="red",main=paste("AUC:",
                                         (perf.rocr@y.values)))
#To get their Roc
theme_set(theme_bw())
levels(result$actual) = c("0", "1")
comb <- unique(result$prediction)
comb <- as.vector(comb)
comb <- comb[-79:-100]
new0 <-c()
new1 <-c()
new2 <-c()
new3 <-c()
new4 <-c()
new5 <-c()
new6 <-c()
matri <- matrix(,nrow = 2,ncol = 2)
for(i in 1:length(comb)) {
  result <- result %>%
    mutate(pred_class = if_else(prediction>= comb[i], "1", "0")) %>%
    mutate_at(vars(actual, pred_class), list(. %>% factor() %>% forcats::fct_relevel("1")))
  
  #To calculate the sensitivity
  new0[i] <-result %>%
    sens(actual, pred_class)%>%
    select(.estimate)
  new1[i] <- result %>%
    spec(actual, pred_class)%>%
    select(.estimate)
  
  #new2[i] <-result %>%
  # spec(actual, pred_class)%>%
  
  #select(.estimate)
  new3[i] <-result %>%
    precision(actual, pred_class)%>%
    select(.estimate)
  new4[i] <-result %>%
    recall(actual, pred_class) %>%
    select(.estimate)
  new5[i] <-result %>%
    accuracy(actual, pred_class) %>%
    select(.estimate)
  new6[i] <-result %>%
    f_meas(actual, pred_class) %>%
    select(.estimate)
  
#Confusion Matrix
mat <- confusionMatrix(result$pred_class, result$actual, positive="1")
#Confusion matrix
matri[i] <- mat$table
}
comb <- as.data.frame(comb)
comb <- t(comb)
new0 <- as.data.frame(new0)
new0 <- t(new0)
new1 <- as.data.frame(new1)
new1 <- t(new1)
new2 <-as.data.frame(new2)
new2 <- t(new2)
new3 <-as.data.frame(new3)
new3 <- t(new3)
new4 <-as.data.frame(new4)
new4 <- t(new4)
new5 <-as.data.frame(new5)
new5 <- t(new5)
new6 <-as.data.frame(new6)
new6 <- t(new6)
neural_compare11 <- cbind(new0,new1,new3,new4,new5,new6)
neural_compare11 <- cbind(comb,neural_compare11)
neural_compare11$comb <- neural_compare11$threshold


colnames(neural_compare11)<-list("threshold","sensitivity","specificity","precision","recall","accuracy","f-measure")
neural_compare11 <- as.data.frame(neural_compare11)
data_out <- write.csv(neural_compare11, "neural_compare1.csv",row.names = FALSE,col.names =
                        TRUE)
roc_dat <- result %>%
  roc_curve(actual, prediction)
roc_dat
pr_dat <- result %>%
  pr_curve(actual, prediction)
pr_dat
roc_dat %>%
  arrange(.threshold) %>% # this step is not strictly necessary here because the rows are already ordered by
  `.threshold`
ggplot() +
  geom_path(aes(1 - specificity, sensitivity)) + # connect the points in the order in which they appear in the data to form a curve
geom_abline(intercept = 0, slope = 1, linetype = "dotted") + # add a reference line by convention
  coord_equal()
pr_dat %>%
  arrange(.threshold) %>% # this step is not strictly necessary here because the rows are already ordered by
  `.threshold`
ggplot() +
  geom_path(aes(recall, precision)) + # connect the points in the order in which they appear in the data to form a curve
coord_equal()
result %>%
  roc_auc(actual, prediction)
result %>%
  pr_auc(actual, prediction)
#This gets the optimal threshold from the list of thresholds
#method 1
threshold1 <- function(predict, response) {
  perf <- ROCR::performance(ROCR::prediction(predict, response), "sens", "spec")
  df <- data.frame(cut = perf@alpha.values[[1]], sens = perf@x.values[[1]], spec = perf@y.values[[1]])
  df[which.max(df$sens + df$spec), "cut"]
}
#Method 2
threshold2 <- function(predict, response) {
  r <- pROC::roc(response, predict)
  r$thresholds[which.max(r$sensitivities + r$specificities)]
}
threshold1(result$prediction, result$actual)
threshold2(result$prediction, result$actual)
#Using the threshold gotten to get sensitivity,specificity et al
result <- result %>%
  #mutate(pred_class = if_else(prediction>= 0.01404586, "1", "0")) %>%
  mutate(pred_class = if_else(prediction>= 0.5, "1", "0")) %>%
  mutate_at(vars(actual, pred_class), list(. %>% factor() %>% forcats::fct_relevel("1")))
#To calculate the sensitivity
result %>%
  sens(actual, pred_class)
result %>%
  spec(actual, pred_class)
result %>%
  spec(actual, pred_class)
result %>%
  precision(actual, pred_class)
result %>%
  recall(actual, pred_class)
result %>%
  accuracy(actual, pred_class)
result %>%
  f_meas(actual, pred_class)
mat <- confusionMatrix(result$pred_class, result$actual, positive="1")
#Confusion matrix
mat$table

