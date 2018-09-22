library(dummies) # dummy.data.frame
library(car) #scatterplotMatrix
library(carData)

# Multivariate linear regression
KC <- read.csv("C:/Users/young/Documents/GitHub/KU-AI/180816/KingCounty.csv")
str(KC)

# 1-C coding
dat <- dummy.data.frame(KC, names=c("waterfront"))
str(dat)

# Indices for the activated input variables
nObs <- dim(KC)[1]
# Split the data into the training/validation sets
set.seed(123)
trn_idx <- sample(1:nObs, round(0.7*nObs))
trn_data <- dat[trn_idx,]
val_data <- dat[-trn_idx,]
dim(trn_data)
dim(val_data)

# Train the MLR
full_model <- lm(price ~ ., data = trn_data)
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

## scatter plot
scatterplotMatrix(~ price + bedrooms + bathrooms + condition +grade + sqft_living + sqft_lot + yr_built,
                  data = trn_data, diagonal="density", reg.line=FALSE, smoother=FALSE)
scatterplotMatrix(~ log(price) + bedrooms + bathrooms + condition +grade + sqft_living + sqft_lot + yr_built,
                  data = trn_data, diagonal="density", reg.line=FALSE, smoother=FALSE)

# Train the MLR
full_model_tfY <- lm(log(price) ~ ., data = trn_data)
summary(full_model_tfY)

# Plot the result
par(mfrow=c(1,3))
#residual plot
plot(fitted(full_model_tfY), residuals(full_model_tfY), xlab="Fitted values", ylab="Residuals", main="residual plot")
abline(0,0,lty=3)

#box plot
boxplot(residuals(full_model_tfY),ylab="Residuals", main="box plot")

#Normal probability plot
qqnorm(rstandard(full_model_tfY), ylab="Standardized residuals", xlab="Normal scores", main="Normal Q-Q")
abline(0,1, col="red")

# prediction
full_fit <- predict(full_model, newdata = val_data)
full_fit_tfY <- predict(full_model_tfY, newdata = val_data)
full_fit_tfY1 <- exp(full_fit_tfY)

# Evaluation index for regression analysis_performance projection
# 1: Mean squared error (MSE)
perf_mat <- matrix(0,4,2)
perf_mat[1,1] <- mean((val_data$price-full_fit)^2)
perf_mat[1,2] <- mean((val_data$price-full_fit_tfY1)^2)

# 2: Root mean squared error (RMSE)
perf_mat[2,1] <- sqrt(mean((val_data$price-full_fit)^2))
perf_mat[2,2] <- sqrt(mean((val_data$price-full_fit_tfY1)^2))

# 3: Mean absolute error (MAE)
perf_mat[3,1] <- mean(abs(val_data$price-full_fit))
perf_mat[3,2] <- mean(abs(val_data$price-full_fit_tfY1))

# 4: Mean absolute percentage error (MAPE)
perf_mat[4,1] <- mean(abs((val_data$price-full_fit)/val_data$price))*100
perf_mat[4,2] <- mean(abs((val_data$price-full_fit_tfY1)/val_data$price))*100

# Result comparison
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("Original", "Transformation_Y")
perf_mat
