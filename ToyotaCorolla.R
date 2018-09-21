###############################ToyotaCorolla#################################

# setting the Working directory 
# 작업할 공간 폴더를 지정해 준다
setwd("C:/Users/young/Desktop/교수실습/8월16일실습(다중회귀)")

# Multivariate linear regression다
# Setting the working directory를 지정해 주면 그냥 파일이름 및 형식만 알려주면 바로 불러와짐 // 혹시나 Setting working directory를 지정해 주지 않을 경우에 절대 경로를 지정해줌
corolla <- read.csv("ToyotaCorolla.csv")
str(corolla)

# Indices for the activated input variables
# ncar: instance or Record의 개수를 확인하는 방법
# nvor: 변수의 갯수를 확인 하는 방법

nCar <- dim(corolla)[1]
nVar <- dim(corolla)[2]

###########################categorical variable ############ 변환

# categorical variable -> 1-of-C coding
# categorical variable 를 one hot encoding 해줌 혹은 그냥 as.factor를 사용하여 바로 할 수 있음.

# 일단 instance 갯수 만큼 3개에 대해서 0을 만들어준다.
dummy_p <- rep(0,nCar) # Petrol
dummy_d <- rep(0,nCar) # Diesel
dummy_c <- rep(0,nCar) # CNG

# 그리고 petrol Diesel CNG에 대해서 어디에 위치하고 있는지 찾아준다.
p_idx <- which(corolla$Fuel_Type == "Petrol")
d_idx <- which(corolla$Fuel_Type == "Diesel")
c_idx <- which(corolla$Fuel_Type == "CNG")

# 그 인덱스에 해당 카테고리컬 변수를 1로 넣어준ㄷ
dummy_p[p_idx] <- 1
dummy_d[d_idx] <- 1
dummy_c[c_idx] <- 1

# 그리고 Fuel에 대해서 데이터 프레임형식으로 만들어 준후 이름을 붙여준다.
Fuel <- data.frame(dummy_p, dummy_d, dummy_c)
names(Fuel) <- c("Petrol","Diesel","CNG")


# 그런 후에 cbind를 통해서 붙여주고 -3 세번째 있는 변수를 빼준다.
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
#residual plot
plot(fitted(full_model), residuals(full_model), xlab="Fitted values", ylab="Residuals", main="residual plot")
abline(0,0,lty=3)

#box plot
boxplot(residuals(full_model),ylab="Residuals", main="box plot")

#Normal probability plot
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


# Upperbound formula 만들기
tmp_x <- paste(colnames(trn_data)[-11], collapse=" + ")
tmp_xy <- paste("Price ~ ", tmp_x, collapse = "")
tmp_xy
as.formula(tmp_xy)

# 변수선택 1: 전진선택법
forward_model <- step(lm(Price ~ 1, data = trn_data), 
                      scope = list(upper = tmp_xy, lower = Price ~ 1), direction="forward", trace=0)
summary(forward_model)


# 변수선택 2: 후진소거법
backward_model <- step(full_model, scope = list(upper = tmp_xy, lower = Price ~ 1), direction="backward", trace=0)
summary(backward_model)


# 변수선택 3: 단계적 선택법
stepwise_model <- step(lm(Price ~ 1, data = trn_data), 
                       scope = list(upper = tmp_xy, lower = Price ~ 1), direction="both", trace=0)
summary(stepwise_model)


# 검증 데이터에 대한 각 변수선택 결과의 예측 정확도 비교
full_fit <- predict(full_model, newdata = val_data)
forward_fit <- predict(forward_model, newdata = val_data)
backward_fit <- predict(backward_model, newdata = val_data)
stepwise_fit <- predict(stepwise_model, newdata = val_data)

# 회귀분석 예측성능 평가지표
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

# 변수선택 기법 결과 비교
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("All", "Forward", "Backward", "Stepwise")
perf_mat
