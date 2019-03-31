library(ggplot2)
library(dplyr)
library(repr)
library(caret)
library(eeptools)    # for DOB to age conversion
library(ROCR)
library(pROC)      # for AUC and ROC 

#1. Extraction of Data
adventure= read.csv('AdvWorksCusts.csv', header= TRUE, stringsAsFactors= TRUE)
avgMonthSpend = read.csv('AW_AveMonthSpend.csv', header= TRUE, stringsAsFactors = FALSE)
bikebuyer= read.csv('AW_BikeBuyer.csv', header= TRUE, stringsAsFactors= FALSE)

# The first 8 columns don't have much effect on the purchasing habit, so will be discarded here.
adventure[,1:8]= NULL
adventure[, 'PhoneNumber'] = NULL

#Checking for duplicates 
dim(distinct(adventure))== dim(adventure)    # no duplicates found

#Checking for any missing values
lapply(adventure, function(x) (any(x=='?')))       #No rows with missing values found.

#Merging the columns from 'avgMonthSpend' and 'bikebuyer'
adventure$AveMonthSpend <- avgMonthSpend$AveMonthSpend
adventure$BikeBuyer <- bikebuyer$BikeBuyer

#DOB to age conversion
adventure$BirthDate <- as.Date(adventure$BirthDate)
adventure$BirthDate <- round(age_calc(adventure$BirthDate, unit='years'))

#Since Postal Code and City have over 270 levels, I removed it. 
factor(adventure$PostalCode)
adventure[,c('PostalCode','City')] = NULL

#Extracting numerical columns:

A <- length(names(adventure)) 
feature <- numeric(A)          #creating zero vector of length equal to the column in 'Credit'
for(i in 1:A){
   feature[i]= ifelse(is.numeric(adventure[,i])==TRUE, 1,NA)
   names(feature)[i]=names(adventure[i])
}


feature <- feature[complete.cases(feature)]
num_cols <- names(feature)
num_cols <- num_cols[1:6]        #Since we are not using the last two columns
      # and the last two columns are for accuracy comparison for label prediction

adventure[,'BikeBuyer'] = ifelse(adventure[,'BikeBuyer']==1, 'sale', 'nosale')
adventure[,'BikeBuyer'] = factor(adventure[,'BikeBuyer'], level=c('sale','nosale'))


#Partition and Scaling:
partition <- createDataPartition(adventure[,'BikeBuyer'], times=1, p=0.75, list=FALSE)
training <- adventure[partition,]
test <- adventure[-partition,]

preProcValues= preProcess(adventure[,num_cols], method= c('center', 'scale'))
training[,num_cols] <- predict(preProcValues, training[,num_cols])
test[,num_cols] <- predict(preProcValues, test[,num_cols])

#Setting up Linear Reg. Model

logistic_mod = glm(BikeBuyer ~ BirthDate + HomeOwnerFlag + NumberCarsOwned + NumberChildrenAtHome+
					TotalChildren + YearlyIncome + BirthDate +Gender + MaritalStatus +
					Education+ CountryRegionName,
			family = quasibinomial,
			data = training)
#logistic_mod = glm(BikeBuyer ~ sum(names(adventure)),
			 family= binomial,
#			data= training)


logistic_mod$coefficients

test$probs = predict(logistic_mod, newdata= test, type = 'response')

threshold = 0.55
test$score = ifelse(test$probs > threshold, 'nosale', 'sale')


## Evaluating the results:
logistic.eval <- function(df){
 #1. Creating a confusion matrix
 df$conf = ifelse(df$BikeBuyer == 'sale' & df$score == 'sale', 'TP',
		 ifelse(df$BikeBuyer == 'sale' & df$score == 'nosale', 'FN',
			ifelse(df$BikeBuyer == 'nosale' & df$score == 'nosale', 'TN', 'FP')))

 #2. Elements of confusion matrix
 TP = length(df[df$conf == 'TP', 'conf'])
 FP = length(df[df$conf == 'FP', 'conf'])
 TN = length(df[df$conf == 'TN', 'conf'])
 FN = length(df[df$conf == 'FN', 'conf'])

 #3. Confusion matrix as a data frame
 out = data.frame(Negative = c(TN, FN), Positive = c(FP, TP))
 row.names(out) = c('Actual Negative', 'Actual Positive')
 print(out)

 #4 . Compute and print metrics
 P = TP/(TP + FP)
 A = (TP + TN)/(TP+TN+FP+FN)
 R = TP/(TP + FN)
 F1 = 2*(P*R)/(P+R)
 S = TN/(TN+FP)
 cat('\n')
 cat(paste('Accuracy    	=', as.character(round(A,3)), '\n'))
 cat(paste('Precision 		=', as.character(round(P,3)), '\n'))
 cat(paste('Recall 		=', as.character(round(R,3)), '\n'))
 cat(paste('F1			=', as.character(round(F1,3)), '\n'))
 cat(paste('Specificity		=', as.character(round(S, 3)), '\n'))

 roc_obj <- roc(df$BikeBuyer, df$probs)
 cat(paste('AUC			=', as.character(round(auc(roc_obj), 3)), '\n'))
}

logistic.eval(test)

#####  Setting up Linear Regression Model for Avg. Monthly Spend:

lin_mod = lm(log_spend ~ CountryRegionName + BirthDate + Education + Gender + 
				Occupation + MaritalStatus + HomeOwnerFlag + NumberCarsOwned +
				NumberChildrenAtHome + TotalChildren + YearlyIncome, 
				data = training)
summary(lin_mod)$coefficients

### Evaluating the Metrics:

print_metrics = function(lin_mod, df, score, label){
 resids = df[, label]-score
 resids2 = resids**2
 N = length(score)
 r2= as.character(round(summary(lin_mod)$r.squared, 4))
 adj_r2 = as.character(round(summary(lin_mod)$adj.r.squared, 4))
 cat(paste("Mean Squared Error	=", as.character(round(sum(resids2)/N,4)),'\n'))
 cat(paste("Root Mean Squared Error =", as.character(round(sqrt(sum(resids2)/N),4)),'\n'))
 cat(paste("Mean Absolute Error 	=", as.character(round(sum(abs(resids2))/N,4)), '\n'))
 cat(paste("Median Absolute Error 	=", as.character(round(median(abs(resids2))/N, 4)), '\n'))
 cat(paste("R^2				=", r2, '\n'))
 cat(paste("adj. R^2			=", adj_r2, '\n'))
}

score = predict(lin_mod, newdata= test)
print_metrics(lin_mod, test, score, label= 'log_spend')

## Retrieving the data set to make predictions on
pred_data = read.csv('AW_test.csv', header=TRUE, stringsAsFactor= TRUE)


##Writing a function to convert DOB, scale and predict monthly spending.
       #Uses some variables from earlier, also the linear_reg model obtained earlier.

feature_names=names(adventure)[1:12]    #taking out label names 

data_predictions = function(inp_data, threshold){
 # DOB conversion to proper format
 inp_data$BirthDate <- as.Date(inp_data$BirthDate, format="%m/%d/%Y")  
 inp_data$BirthDate <- round(age_calc(inp_data$BirthDate, unit='years'))
 str(inp_data$BirthDate)
 cat('\n')
 
 #Data normalization
 z_score_transform <- preProcess(inp_data[, feature_names], method= c('center', 'scale'))
 inp_data[, feature_names] <- predict(z_score_transform, inp_data[,feature_names])
 
 ## Calling out the Linear regression model from earlier to predict the avg. spending
 avg_spend = predict(lin_mod, newdata= inp_data)
 cat(paste('The log value of average monthly spending: '),'\n')
 print(head(avg_spend))
 
 avg_spend= round(exp(avg_spend),2)
 cat('\n')
 cat(paste('The Average spending after log transformation is:'),'\n')
 print(avg_spend)         # Printing out sample output
 
 ## Merging the values to the data frame
 inp_data$AveSpending = avg_spend
 
 # Calling out Logistic Model to predict BikeBuyers
 inp_data$BikeBuyer= predict(logistic_mod, inp_data)
 inp_data$BikeBuyer= ifelse(inp_data$BikeBuyer > threshold, 'sale', 'nosale')


 write.csv(inp_data, 'Prediction_Values.csv')
}

data_predictions(head(pred_data), threshold= 0.55)     #use this to get quick sample output 
							#using 5 rows of datapoints.

#data_predictions(pred_data, threshold= 0.55)       

viewPred= read.csv("Prediction_Values.csv", header= TRUE, stringsAsFactors= TRUE)
head(viewPred)














