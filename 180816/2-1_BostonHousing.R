#######################BostonHousing########################################
# setting the Working directory 
setwd("C:/Users/young/Documents/GitHub/KU-AI/180816")

# Multivariate linear regression
boston <- read.csv("BostonHousing.csv")
str(boston)

# Indices for the activated input variables
nObs <- dim(boston)[1]

# Split the data into the training/validation sets
trn_idx <- sample(1:nObs, round(0.7*nObs))
trn_data <- boston[trn_idx,]
val_data <- boston[-trn_idx,]
dim(trn_data)
dim(val_data)

# Train the MLR
full_model <- lm(MEDV ~ ., data = trn_data)
summary(full_model)

# Plot the result
par(mfrow=c(1,3))

#residual plot
plot(fitted(full_model), residuals(full_model), xlab="Fitted values", ylab="Residuals", main="residual plot")
abline(0,0,lty=3)

#box plot
boxplot(residuals(full_model),ylab="Residuals", main="box plot")

#Normal probability plot
qqnorm(rstandard(full_model), ylab="Standardized residuals", xlab="Normal scores", main="Normal Q-Q")
abline(0,1, col="red")

dev.off()

# fitted value & real value
plot(trn_data$MEDV, fitted(full_model), xlab="real value", ylab="fitted value")
abline(0,1,lty=3)

# prediction
full_fit <- predict(full_model, newdata = val_data)

matplot(cbind(full_fit, val_data$MEDV),pch=1, type="b", col=c(1:2), ylab="", xlab="observation")
legend("topright", legend=c("fitted value", "real value"), pch=1, col=c(1:2), bg="white")

# To make Upperbound formula
tmp_x <- paste(colnames(trn_data)[-12], collapse=" + ")
tmp_xy <- paste("MEDV ~ ", tmp_x, collapse = "")
as.formula(tmp_xy)

# variable choose 1: forward selection method
forward_model <- step(lm(MEDV ~ 1, data = trn_data), 
                      scope = list(upper = tmp_xy, lower = MEDV ~ 1), direction="forward", trace=0)
summary(forward_model)

# variable choose 2: backward elimination method
backward_model <- step(full_model, scope = list(upper = tmp_xy, lower = MEDV ~ 1), direction="backward", trace=0)
summary(backward_model)

# variable choose 3: stepwise selection method
stepwise_model <- step(lm(MEDV ~ 1, data = trn_data), 
                       scope = list(upper = tmp_xy, lower = MEDV ~ 1), direction="both", trace=0)
summary(stepwise_model)

# Comparison of prediction accuracy of each variable selection result on verification data
full_fit <- predict(full_model, newdata = val_data)
forward_fit <- predict(forward_model, newdata = val_data)
backward_fit <- predict(backward_model, newdata = val_data)
stepwise_fit <- predict(stepwise_model, newdata = val_data)

# Evaluation index for regression analysis performance projection
# 1: Mean squared error (MSE)
perf_mat <- matrix(0,4,4)
perf_mat[1,1] <- mean((val_data$MEDV-full_fit)^2)
perf_mat[1,2] <- mean((val_data$MEDV-forward_fit)^2)
perf_mat[1,3] <- mean((val_data$MEDV-backward_fit)^2)
perf_mat[1,4] <- mean((val_data$MEDV-stepwise_fit)^2)

# 2: Root mean squared error (RMSE)
perf_mat[2,1] <- sqrt(mean((val_data$MEDV-full_fit)^2))
perf_mat[2,2] <- sqrt(mean((val_data$MEDV-forward_fit)^2))
perf_mat[2,3] <- sqrt(mean((val_data$MEDV-backward_fit)^2))
perf_mat[2,4] <- sqrt(mean((val_data$MEDV-stepwise_fit)^2))

# 3: Mean absolute error (MAE)
perf_mat[3,1] <- mean(abs(val_data$MEDV-full_fit))
perf_mat[3,2] <- mean(abs(val_data$MEDV-forward_fit))
perf_mat[3,3] <- mean(abs(val_data$MEDV-backward_fit))
perf_mat[3,4] <- mean(abs(val_data$MEDV-stepwise_fit))

# 4: Mean absolute percentage error (MAPE)
perf_mat[4,1] <- mean(abs((val_data$MEDV-full_fit)/val_data$MEDV))*100
perf_mat[4,2] <- mean(abs((val_data$MEDV-forward_fit)/val_data$MEDV))*100
perf_mat[4,3] <- mean(abs((val_data$MEDV-backward_fit)/val_data$MEDV))*100
perf_mat[4,4] <- mean(abs((val_data$MEDV-stepwise_fit)/val_data$MEDV))*100

# comparison of result for variable selection technique
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("All", "Forward", "Backward", "Stepwise")
perf_mat

# Variable change
pairs(boston)

par(mfrow=c(1,2))
plot(boston$MEDV, boston$LSTAT)
plot(boston$MEDV, log(boston$LSTAT))

boston_new<-boston
boston_new$LSTAT<-log(boston$LSTAT)

#split
trn_data_new <- boston_new[trn_idx,]
val_data_new <- boston_new[-trn_idx,]

# Train the MLR
full_model_new <- lm(MEDV ~ ., data = trn_data_new)
summary(full_model_new)