# Multivariate linear regression
cold <- read.csv("C:/Users/young/Documents/GitHub/KU-AI/180816/cold.csv")
str(cold)

# Indices for the activated input variables
nObs <- dim(cold)[1]
nVar <- dim(cold)[2]

# categorical variable -> 1-of-C coding
dummy_1 <- rep(0,nObs)
dummy_2 <- rep(0,nObs)
dummy_3 <- rep(0,nObs)
dummy_4 <- rep(0,nObs)
dummy_5 <- rep(0,nObs)
dummy_6 <- rep(0,nObs)
dummy_7 <- rep(0,nObs)

idx1 <- which(cold$요일 == "월요일")
idx2 <- which(cold$요일 == "화요일")
idx3 <- which(cold$요일 == "수요일")
idx4 <- which(cold$요일 == "목요일")
idx5 <- which(cold$요일 == "금요일")
idx6 <- which(cold$요일 == "토요일")
idx7 <- which(cold$요일 == "일요일")

dummy_1[idx1] <- 1
dummy_2[idx2] <- 1
dummy_3[idx3] <- 1
dummy_4[idx4] <- 1
dummy_5[idx5] <- 1
dummy_6[idx6] <- 1
dummy_7[idx7] <- 1

day <- data.frame(dummy_1, dummy_2, dummy_3, dummy_4, dummy_5, dummy_6, dummy_7)
names(day) <- c("월","화","수","목","금","토","일")

# Prepare the data for MLR
mlr_data <- cbind(cold[,-17],day)
str(mlr_data)

# Split the data into the training/validation sets
trn_idx <- sample(1:nObs, round(0.7*nObs))
trn_data <- mlr_data[trn_idx,]
val_data <- mlr_data[-trn_idx,]

# Train the MLR
full_model <- lm(진료건수 ~ ., data = trn_data)
summary(full_model)

# Plot the result
#residual plot
plot(fitted(full_model), residuals(full_model), xlab="Fitted values", ylab="Residuals", main='residual plot')
abline(0,0,lty=3)

#box plot
boxplot(residuals(full_model),ylab="Residuals", main="box plot")

#Normal probability plot
qqnorm(rstandard(full_model), ylab="Standardized residuals", xlab="Normal scores", main="Normal Q-Q")
abline(0,1, col="red")

# fitted value & real value
plot(trn_data$진료건수, fitted(full_model), xlab="real value", ylab="fitted value")
abline(0,1,lty=3)

# prediction
full_fit <- predict(full_model, newdata = val_data)
matplot(cbind(full_fit, val_data$진료건수),pch=1, type="b", col=c(1:2), ylab="", xlab="observation")
legend("topright", legend=c("fitted value", "real value"), pch=1, col=c(1:2), bg="white")

# To make upperbound formula
tmp_x <- paste(colnames(trn_data)[-1], collapse=" + ")
tmp_xy <- paste("진료건수 ~ ", tmp_x, collapse = "")
tmp_xy
as.formula(tmp_xy)


# variable choose 1: forward selection method
forward_model <- step(lm(진료건수 ~ 1, data = trn_data), 
                      scope = list(upper = tmp_xy, lower = 진료건수 ~ 1), direction="forward", trace=0)
summary(forward_model)


# variable choose 2: backward elimination method
backward_model <- step(full_model, scope = list(upper = tmp_xy, lower = 진료건수 ~ 1), direction="backward", trace=0)
summary(backward_model)


# variable choose 3: stepwise selection method
stepwise_model <- step(lm(진료건수 ~ 1, data = trn_data), 
                       scope = list(upper = tmp_xy, lower = 진료건수 ~ 1), direction="both", trace=0)
summary(stepwise_model)


# # Comparison of prediction accuracy of each variable selection result on verification data
full_fit <- predict(full_model, newdata = val_data)
forward_fit <- predict(forward_model, newdata = val_data)
backward_fit <- predict(backward_model, newdata = val_data)
stepwise_fit <- predict(stepwise_model, newdata = val_data)

# Evaluation index for regression analysis performance projection
# 1: Mean squared error (MSE)
perf_mat <- matrix(0,4,4)
perf_mat[1,1] <- mean((val_data$진료건수-full_fit)^2)
perf_mat[1,2] <- mean((val_data$진료건수-forward_fit)^2)
perf_mat[1,3] <- mean((val_data$진료건수-backward_fit)^2)
perf_mat[1,4] <- mean((val_data$진료건수-stepwise_fit)^2)

# 2: Root mean squared error (RMSE)
perf_mat[2,1] <- sqrt(mean((val_data$진료건수-full_fit)^2))
perf_mat[2,2] <- sqrt(mean((val_data$진료건수-forward_fit)^2))
perf_mat[2,3] <- sqrt(mean((val_data$진료건수-backward_fit)^2))
perf_mat[2,4] <- sqrt(mean((val_data$진료건수-stepwise_fit)^2))

# 3: Mean absolute error (MAE)
perf_mat[3,1] <- mean(abs(val_data$진료건수-full_fit))
perf_mat[3,2] <- mean(abs(val_data$진료건수-forward_fit))
perf_mat[3,3] <- mean(abs(val_data$진료건수-backward_fit))
perf_mat[3,4] <- mean(abs(val_data$진료건수-stepwise_fit))

# 4: Mean absolute percentage error (MAPE)
perf_mat[4,1] <- mean(abs((val_data$진료건수-full_fit)/val_data$진료건수))*100
perf_mat[4,2] <- mean(abs((val_data$진료건수-forward_fit)/val_data$진료건수))*100
perf_mat[4,3] <- mean(abs((val_data$진료건수-backward_fit)/val_data$진료건수))*100
perf_mat[4,4] <- mean(abs((val_data$진료건수-stepwise_fit)/val_data$진료건수))*100

# comparison of result for variable selection technique
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("All", "Forward", "Backward", "Stepwise")
perf_mat
