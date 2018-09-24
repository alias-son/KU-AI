###############################ToyotaCorolla#################################

# setting the Working directory 
setwd("C:/Users/young/Documents/GitHub/KU-AI/180816")

# Multivariate linear regression
# Setting the working directory
corolla <- read.csv("ToyotaCorolla.csv")
str(corolla)

# Indices for the activated input variables
# ncar: How to check the number of instance or Record
# nvor: How to confirm the figure for variable

nCar <- dim(corolla)[1]
nVar <- dim(corolla)[2]

###########################categorical variable###########################

# categorical variable -> 1-of-C coding
# categorical variable to one hot encoding Or use as.factor

# Make 0 for the number of instances, petrol, diesel and cng
dummy_p <- rep(0,nCar) # Petrol
dummy_d <- rep(0,nCar) # Diesel
dummy_c <- rep(0,nCar) # CNG

# fine the postion of instances
p_idx <- which(corolla$Fuel_Type == "Petrol")
d_idx <- which(corolla$Fuel_Type == "Diesel")
c_idx <- which(corolla$Fuel_Type == "CNG")

# To input categorical variables as 1 in the index
dummy_p[p_idx] <- 1
dummy_d[d_idx] <- 1
dummy_c[c_idx] <- 1

# change form as data.frame then nameing
Fuel <- data.frame(dummy_p, dummy_d, dummy_c)
names(Fuel) <- c("Petrol","Diesel","CNG")


# use cbind to attach it and then subtract -3 variable.
mlr_data <- cbind(Fuel, corolla[,-3])
str(mlr_data)

# Split the data into the training/validation sets
trn_idx <- sample(1:nCar, round(0.7*nCar))
trn_data <- mlr_data[trn_idx,]
val_data <- mlr_data[-trn_idx,]

dim(trn_data)
dim(val_data)

# Train the MLR
full_model <- lm(Price ~ ., data = trn_data)
summary(full_model)

# Plot the result
# residual plot
plot(fitted(full_model), residuals(full_model), xlab="Fitted values", ylab="Residuals", main="residual plot")
abline(0,0,lty=3)

# box plot
boxplot(residuals(full_model),ylab="Residuals", main="box plot")

# Normal probability plot
qqnorm(rstandard(full_model), ylab="Standardized residuals", xlab="Normal scores", main="Normal Q-Q")
abline(0,1, col="red")

# fitted value & real value
plot(trn_data$Price, fitted(full_model), xlim = c(4000,35000), ylim = c(4000,30000),
     xlab="real value(Price)", ylab="fitted value")
abline(0,1,lty=3)

options(warn = -1)

# prediction
full_fit <- predict(full_model, newdata = val_data)

matplot(cbind(full_fit, val_data$Price),pch=1, type="b", col=c(1:2), ylab="", xlab="observation")
legend("topright", legend=c("fitted value", "real value"), pch=1, col=c(1:2), bg="white")


# To make Upperbound formula
tmp_x <- paste(colnames(trn_data)[-11], collapse=" + ")
tmp_xy <- paste("Price ~ ", tmp_x, collapse = "")
tmp_xy
as.formula(tmp_xy)

# variable choose 1: forward selection method
forward_model <- step(lm(Price ~ 1, data = trn_data), 
                      scope = list(upper = tmp_xy, lower = Price ~ 1), direction="forward", trace=0)
summary(forward_model)


# variable choose 2: backward elimination method
backward_model <- step(full_model, scope = list(upper = tmp_xy, lower = Price ~ 1), direction="backward", trace=0)
summary(backward_model)


# variable choose 3: stepwise selection method
stepwise_model <- step(lm(Price ~ 1, data = trn_data), 
                       scope = list(upper = tmp_xy, lower = Price ~ 1), direction="both", trace=0)
summary(stepwise_model)


# Comparison of prediction accuracy of each variable selection result on verification data
full_fit <- predict(full_model, newdata = val_data)
forward_fit <- predict(forward_model, newdata = val_data)
backward_fit <- predict(backward_model, newdata = val_data)
stepwise_fit <- predict(stepwise_model, newdata = val_data)

# Evaluation index for regression analysis performance projection
# 1: Mean squared error (MSE)
perf_mat <- matrix(0,4,4)
perf_mat[1,1] <- mean((val_data$Price-full_fit)^2)
perf_mat[1,2] <- mean((val_data$Price-forward_fit)^2)
perf_mat[1,3] <- mean((val_data$Price-backward_fit)^2)
perf_mat[1,4] <- mean((val_data$Price-stepwise_fit)^2)

# 2: Root mean squared error (RMSE)
perf_mat[2,1] <- sqrt(mean((val_data$Price-full_fit)^2))
perf_mat[2,2] <- sqrt(mean((val_data$Price-forward_fit)^2))
perf_mat[2,3] <- sqrt(mean((val_data$Price-backward_fit)^2))
perf_mat[2,4] <- sqrt(mean((val_data$Price-stepwise_fit)^2))

# 3: Mean absolute error (MAE)
perf_mat[3,1] <- mean(abs(val_data$Price-full_fit))
perf_mat[3,2] <- mean(abs(val_data$Price-forward_fit))
perf_mat[3,3] <- mean(abs(val_data$Price-backward_fit))
perf_mat[3,4] <- mean(abs(val_data$Price-stepwise_fit))

# 4: Mean absolute percentage error (MAPE)
perf_mat[4,1] <- mean(abs((val_data$Price-full_fit)/val_data$Price))*100
perf_mat[4,2] <- mean(abs((val_data$Price-forward_fit)/val_data$Price))*100
perf_mat[4,3] <- mean(abs((val_data$Price-backward_fit)/val_data$Price))*100
perf_mat[4,4] <- mean(abs((val_data$Price-stepwise_fit)/val_data$Price))*100

# comparison of result for variable selection technique
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("All", "Forward", "Backward", "Stepwise")
perf_mat
