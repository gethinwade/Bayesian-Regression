
library(dplyr)
library(tidyr)
library(gt)
library(corrplot)
library(rstan)
library(HDInterval)
library(dbarts)
library(caret)


# =========================================
# Data Import, Cleaning, and Initial Checks
# =========================================

# Importing dataset
df <- read.csv("kc_house_data.csv")

# Dropping non-predictive variables
df <- df %>%
    select(-c(lat, long, date, yr_renovated, sqft_above, sqft_basement, floors, id, zipcode))

# Creating variable 'age' and dropping yr_built 
df <- df %>%
    mutate(age = 2015 - yr_built) %>%
    select(-yr_built)

# Checking dimensions and for missing values
cat("Dimensions:", nrow(df), "x", ncol(df), "\n\n")
cat("Missing values per column:\n")
print(colSums(is.na(df)))
cat("\nTotal missing:", sum(is.na(df)), "\n")


# =========================
# Exploratory Data Analysis
# =========================

# ========== Summary Statistics ========== #

continuous_vars <- c("bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "age", "sqft_living15", "sqft_lot15")

df %>%
    summarise(across(all_of(continuous_vars),
        list(Median = median, Mean = mean, SD = sd, Min = min, Max = max))) %>%
    pivot_longer(everything(), names_to = c("Variable", "Statistic"),
        names_pattern = "(.+)_(Median|Mean|SD|Min|Max)$", values_to = "Value") %>%
    pivot_wider(names_from = Statistic, values_from = Value) %>%
    gt() %>%
    tab_header(title = "Descriptive Statistics") %>%
    fmt_number(columns = c(Median, Mean, SD, Min, Max), decimals = 2)


# ========== Exploratory Plots ========== #

par(mfrow = c(2, 2))

# Histogram of Price
hist(df$price, main = "Distribution of Home Price", xlab = "Price",
    col = "grey20", border = "white", breaks = 50)

# Square Footage vs. Price by Waterfront Plot
plot(df$sqft_living, df$price, main = "Home Price by Living Area",
    xlab = "sqft_living", ylab = "Price", pch = 16, cex = 0.6,
    col = ifelse(df$waterfront == 1, "green", "blue"))
legend("topleft", legend = c("Waterfront", "Non-Waterfront"),
    col = c("green", "blue"), pch = 16, cex = 0.8, bty = "n")

# Condition vs. Price Plot
plot(jitter(df$condition, amount = 0.2), df$price, main = "Home Price by Condition",
    xlab = "Condition", ylab = "Price", pch = 16, col = rgb(0, 0, 0, 0.2), cex = 0.6)

# Grade vs. Price Plot
plot(jitter(df$grade, amount = 0.2), df$price, main = "Home Price by Grade",
    xlab = "Grade", ylab = "Price", pch = 16, col = rgb(0, 0, 0, 0.2), cex = 0.6)

par(mfrow = c(1, 1))


# ========== Correlation Matrix ========== #

continuous_vars_corr <- c("price", "sqft_living", "sqft_lot", "sqft_living15", 
    "sqft_lot15", "age", "bedrooms", "bathrooms", "condition", "grade", "waterfront")

corrplot(cor(df[, continuous_vars_corr], use = "complete.obs"),
    method = "color", type = "upper", diag = FALSE, tl.col = "black",
    addCoef.col = "black", number.cex = 0.6, tl.cex = 0.7)


# =======================
# OLS Baseline Regression
# =======================

# ========== Raw OLS Regression ========== #

raw_ols <- lm(price ~ sqft_living + sqft_lot + sqft_living15 + sqft_lot15 +
    age + bedrooms + bathrooms + condition + grade + waterfront, data = df)

summary(raw_ols)


# ========== Raw OLS Residual Plots ========== #

par(mfrow = c(2, 2))

# QQ Plot
qqnorm(residuals(raw_ols), main = "Q-Q Plot of Residuals", pch = 16, cex = 0.4)
qqline(residuals(raw_ols), col = "red")

# Residual Histogram
hist(residuals(raw_ols), main = "Residual Histogram", xlab = "Residuals",
    col = "grey20", border = "white", breaks = 50)

# Residuals vs. Fitted
plot(fitted(raw_ols), residuals(raw_ols), main = "Residuals vs. Fitted",
    xlab = "Fitted Values", ylab = "Residuals", pch = 16, cex = 0.4)
abline(h = 0, col = "red")

# Residual Time Series
plot(residuals(raw_ols), main = "Residual Time Series", xlab = "Index",
    ylab = "Residuals", type = "l")
abline(h = 0, col = "red")

par(mfrow = c(1, 1))


# ========== Box-Cox Transformation ========== #

bc <- MASS::boxcox(raw_ols, data = as.data.frame(df))
best_lambda <- bc$x[which.max(bc$y)]
cat("Best lambda:", best_lambda, "\n")


# ========== Log OLS Regression ========== #

df$log_price <- log(df$price)

# OLS with log price
log_ols <- lm(log_price ~ sqft_living + sqft_lot + sqft_living15 + sqft_lot15 +
    age + bedrooms + bathrooms + condition + grade + waterfront,
    data = df)

summary(log_ols)


# ========== Log OLS Residual Plots ========== #

par(mfrow = c(2, 2))

# QQ Plot
qqnorm(residuals(log_ols), main = "Q-Q Plot of Residuals", pch = 16, cex = 0.4)
qqline(residuals(log_ols), col = "red")

# Residual Histogram
hist(residuals(log_ols), main = "Residual Histogram", xlab = "Residuals",
    col = "grey20", border = "white", breaks = 50)

# Residuals vs. Fitted
plot(fitted(log_ols), residuals(log_ols), main = "Residuals vs. Fitted",
    xlab = "Fitted Values", ylab = "Residuals", pch = 16, cex = 0.4)
abline(h = 0, col = "red")

# Residual Time Series
plot(residuals(log_ols), main = "Residual Time Series", xlab = "Index",
    ylab = "Residuals", type = "l")
abline(h = 0, col = "red")

par(mfrow = c(1, 1))


# ============================
# Bayesian Multiple Regression
# ============================

# ========== Weakly Informative Priors (Normal & Half-Normal) ========== #

stan_model_weak <- "
data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n, p] X;
  vector[n] y;
}

parameters {
  vector[p] beta;
  real alpha;
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);

  // Likelihood
  y ~ normal(alpha + X * beta, sigma);
}
"

# Design Matrix
X <- model.matrix(log_ols)[, -1]
X_scaled <- scale(X)
X_center <- attr(X_scaled, "scaled:center")
X_scale <- attr(X_scaled, "scaled:scale")
y <- df$log_price
n <- nrow(X_scaled)
p <- ncol(X)

# Data List
stan_data_weak <- list(n = n, p = p, X = X_scaled, y = y)

# Fit Model
fit_weak <- rstan::sampling(rstan::stan_model(model_code = stan_model_weak),
    data = stan_data_weak, chains = 4, iter = 3500, warmup = 1000, seed = 42)


# ========== Results with Weakly Informative Priors ========== #

colnames(X)
posterior <- rstan::extract(fit_weak)
print(fit_weak, pars = c("alpha", "beta", "sigma"))

# Trace Plots
rstan::traceplot(fit_weak, pars = c("alpha", "beta", "sigma"))

par(mfrow = c(3, 4))
# Alpha Plot
plot(density(posterior$alpha), main = "Alpha",
    xlab = expression(alpha), ylab = "Density", col = "blue", lwd = 2)
lines(x_alpha, dnorm(x_alpha, 0, 100), col = "red", lwd = 2, lty = 2)

# Beta plots
beta_names <- colnames(X)
x_beta <- seq(-4, 4, length.out = 1000)

for (j in 1:10) {
    plot(density(posterior$beta[, j]),
        main = paste0("Beta ", j, ": ", beta_names[j]),
        xlab = expression(beta), ylab = "Density",
        col = "blue", lwd = 2)
    lines(x_beta, dnorm(x_beta, 0, 1), col = "red", lwd = 2, lty = 2)
}

# Sigma Plot
plot(density(posterior$sigma), main = "Sigma",
    xlab = expression(sigma), ylab = "Density", col = "blue", lwd = 2)
lines(x_sigma, dnorm(x_sigma, 0, 1) * 2, col = "red", lwd = 2, lty = 2)
par(mfrow = c(1, 1))


# Extract posterior beta samples
beta_samples <- posterior$beta

# Compute means and HPDIs
beta_means <- colMeans(beta_samples)
beta_hpdi <- apply(beta_samples, 2, function(x) hdi(x, credMass = 0.95))

# Build data frame
beta_table <- data.frame(
  Parameter = paste0("Beta ", 1:10, ": ", colnames(X)),
  Mean = round(beta_means, 4),
  HPDI_95 = paste0("[", round(beta_hpdi[1, ], 4), ", ", round(beta_hpdi[2, ], 4), "]")
)

# Render as plot
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1),
     axes = FALSE, xlab = "", ylab = "", main = "Beta Coefficients (Weakly Informative)")

text(0.05, 0.97, "Parameter", adj = 0, cex = 0.9, font = 2)
text(0.35, 0.97, "Mean", adj = 0, cex = 0.9, font = 2)
text(0.50, 0.97, "95% HPDI", adj = 0, cex = 0.9, font = 2)
segments(0, 0.94, 0.85, 0.94)

for (i in 1:nrow(beta_table)) {
  text(0.05, 1 - i * 0.09, beta_table$Parameter[i], adj = 0, cex = 0.8)
  text(0.35, 1 - i * 0.09, beta_table$Mean[i], adj = 0, cex = 0.8)
  text(0.5, 1 - i * 0.09, beta_table$HPDI_95[i], adj = 0, cex = 0.8)
}


# ========== Informative Priors (OLS Estimates) ========== #

# Beta prior parameters
beta_ols <- coef(log_ols)[-1]
se_ols <- summary(log_ols)$coefficients[-1, 2]
beta_ols_scaled <- beta_ols * X_scale
se_ols_scaled <- se_ols * X_scale

# Alpha prior parameters
alpha_ols <- coef(log_ols)[1]
se_alpha_ols <- summary(log_ols)$coefficients[1, 2]
alpha_prior_mean_scaled <- alpha_ols + sum(beta_ols * X_center)

# Sigma prior parameters
sigma_ols <- summary(log_ols)$sigma
se_sigma_ols <- sigma_ols / sqrt(2 * (nrow(X) - ncol(X) - 1))

stan_model_informative <- "
data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n, p] X;
  vector[n] y;
  vector[p] beta_prior_mean;
  vector[p] beta_prior_sd;
  real alpha_prior_mean;
  real<lower=0> alpha_prior_sd;
  real<lower=0> sigma_prior_mean;
  real<lower=0> sigma_prior_sd;
}

parameters {
  vector[p] beta;
  real alpha;
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(alpha_prior_mean, alpha_prior_sd);
  beta ~ normal(beta_prior_mean, beta_prior_sd);
  sigma ~ normal(sigma_prior_mean, sigma_prior_sd);

  // Likelihood
  y ~ normal(alpha + X * beta, sigma);
}
"

# Data List
stan_data_informative <- list(n = n, p = p, X = X_scaled, y = y,
    beta_prior_mean = beta_ols_scaled,
    beta_prior_sd = se_ols_scaled,
    alpha_prior_mean = alpha_prior_mean_scaled,
    alpha_prior_sd = se_alpha_ols,
    sigma_prior_mean = sigma_ols,
    sigma_prior_sd = se_sigma_ols
)

# Fit Model
fit_informative <- rstan::sampling(rstan::stan_model(model_code = stan_model_informative),
    data = stan_data_informative, chains = 4, iter = 3500, warmup = 1000, seed = 42)


# ========== Results with Informative Priors ========== #

posterior_inf <- rstan::extract(fit_informative)
print(fit_informative, pars = c("alpha", "beta", "sigma"))

# Trace Plots
rstan::traceplot(fit_informative, pars = c("alpha", "beta", "sigma"))


par(mfrow = c(3, 4))
# Alpha PLot
x_alpha_inf <- seq(alpha_prior_mean_scaled - 4 * se_alpha_ols,
    alpha_prior_mean_scaled + 4 * se_alpha_ols, length.out = 1000)

plot(density(posterior_inf$alpha), main = "Alpha",
    xlab = expression(alpha), ylab = "Density", col = "blue", lwd = 2,
    xlim = c(alpha_prior_mean_scaled - 4 * se_alpha_ols,
    alpha_prior_mean_scaled + 4 * se_alpha_ols))
lines(x_alpha_inf, dnorm(x_alpha_inf, alpha_prior_mean_scaled, se_alpha_ols),
    col = "red", lwd = 2, lty = 2)

# Beta Plots
for (j in 1:10) {
    x_beta_inf <- seq(beta_ols_scaled[j] - 4 * se_ols_scaled[j],
        beta_ols_scaled[j] + 4 * se_ols_scaled[j], length.out = 1000)
    plot(density(posterior_inf$beta[, j]),
        main = paste0("Beta ", j, ": ", beta_names[j]),
        xlab = expression(beta), ylab = "Density", col = "blue", lwd = 2)
  lines(x_beta_inf, dnorm(x_beta_inf, beta_ols_scaled[j], se_ols_scaled[j]),
        col = "red", lwd = 2, lty = 2)
}

# Sigma Plot
x_sigma_inf <- seq(0, sigma_ols + 4 * se_sigma_ols, length.out = 1000)

plot(density(posterior_inf$sigma), main = "Sigma",
    xlab = expression(sigma), ylab = "Density", col = "blue", lwd = 2)
lines(x_sigma_inf, dnorm(x_sigma_inf, sigma_ols, se_sigma_ols),
    col = "red", lwd = 2, lty = 2)
par(mfrow = c(1, 1))


# Extract posterior beta samples
beta_samples_inf <- posterior_inf$beta

# Compute means and HPDIs
beta_means_inf <- colMeans(beta_samples_inf)
beta_hpdi_inf <- apply(beta_samples_inf, 2, function(x) hdi(x, credMass = 0.95))

# Build data frame
beta_table_inf <- data.frame(
  Parameter = paste0("Beta ", 1:10, ": ", colnames(X)),
  Mean = round(beta_means_inf, 4),
  HPDI_95 = paste0("[", round(beta_hpdi_inf[1, ], 4), ", ", round(beta_hpdi_inf[2, ], 4), "]")
)

# Render as plot
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1),
     axes = FALSE, xlab = "", ylab = "")
mtext("Beta Coefficients (Informative Prior)", side = 3,
      line = 1, cex = 1.2, font = 2, adj = 0.5)

text(0.05, 0.97, "Parameter", adj = 0, cex = 0.9, font = 2)
text(0.35, 0.97, "Mean", adj = 0, cex = 0.9, font = 2)
text(0.50, 0.97, "95% HPDI", adj = 0, cex = 0.9, font = 2)
segments(0, 0.94, 0.85, 0.94)

for (i in 1:nrow(beta_table_inf)) {
  text(0.05, 1 - i * 0.09, beta_table_inf$Parameter[i], adj = 0, cex = 0.8)
  text(0.35, 1 - i * 0.09, beta_table_inf$Mean[i], adj = 0, cex = 0.8)
  text(0.50, 1 - i * 0.09, beta_table_inf$HPDI_95[i], adj = 0, cex = 0.8)
}


# ========== Plot of the Final Fitted Model ========== #

# Posterior predictive mean for each observation
y_pred <- apply(posterior_inf$beta, 1, function(b) X_scaled %*% b) + 
    matrix(posterior_inf$alpha, nrow = nrow(X_scaled), ncol = length(posterior_inf$alpha), byrow = TRUE)

# Compute mean and 95% interval across posterior draws
y_pred_mean <- rowMeans(y_pred)
y_pred_lower <- apply(y_pred, 1, quantile, 0.025)
y_pred_upper <- apply(y_pred, 1, quantile, 0.975)

# Plot
plot(y, y_pred_mean, pch = 16, cex = 0.4, col = rgb(0, 0, 1, 0.3),
    main = "Predicted vs. Actual Log Price",
    xlab = "Actual log(Price)", ylab = "Predicted log(Price)")
abline(0, 1, col = "red", lwd = 2)


# ================================ 
#           BART MODEL
# ================================ 

set.seed(27)

# Variables to match those used in Stan models
X_bart = as.matrix(X_scaled)
y_bart = as.vector(y)

# ========== Cross Validation Over base Values ========== #
base_grid = c(0.25, 0.50, 0.75, 0.95)
k = 5
folds = createFolds(y_bart, k = k, list = TRUE)

cv_result = data.frame(base = base_grid, rmse = NA)

for (b in seq_along(base_grid)){
  fold_rmse = numeric(k)

  for (i in 1:k){
    test_idx = folds[[i]]

    X_train = X_bart[-test_idx, ]
    X_test = X_bart[test_idx, ]
    y_train = y_bart[-test_idx]
    y_test = y_bart[test_idx]

    fit_cv = bart(x.train = X_train,
                  y.train = y_train,
                  x.test = X_test,
                  ntree = 200,
                  ndpost = 300,
                  nskip = 150,
                  base = base_grid[b],
                  power = 2.0,
                  verbose = FALSE)
    fold_rmse[i] = sqrt(mean((y_test - fit_cv$yhat.test.mean)^2))
  }

  cv_result$rmse[b] = mean(fold_rmse)
  cat("base = ", base_grid[b], "| CV RMSE =", round(cv_result$rmse[b], 4), "\n")
}

# Best base
best_base = cv_result$base[which.min(cv_result$rmse)]
cat("\nBest base:", best_base, "\n")

# ========== CV Results Plot ========== #
plot(cv_result$base, cv_result$rmse, type = "b",
     pch = 16, col = "blue", lwd = 2,
     main = "Cross-Validation RMSE by Base Parameter",
     xlab = "Base (alpha)", ylab = "CV RMSE")
abline(v = best_base, col = "red", lty = 2)


# ============================================================= 
#            Final BART Model (best base, full data) 
# ============================================================= 

bart_final <- bart(x.train = X_bart,
                   y.train = y_bart,
                   ntree   = 200,
                   ndpost  = 1000,
                   nskip   = 500,
                   base    = best_base,
                   power   = 2.0,
                   keeptrees = TRUE,
                   verbose = FALSE)

# ========== Fitted vs. Actual Plot ========== #

y_pred_bart <- bart_final$yhat.train.mean

plot(y_bart, y_pred_bart,
     pch = 16, cex = 0.4, col = rgb(0, 0, 1, 0.3),
     main = "Predicted vs. Actual Log Price (BART)",
     xlab = "Actual log(Price)", ylab = "Predicted log(Price)")
abline(0, 1, col = "red", lwd = 2)

# ========== Variable Importance ========== #

var_importance <- sort(colMeans(bart_final$varcount), decreasing = TRUE)
var_importance_pct <- round(100 * var_importance / sum(var_importance), 2)

barplot(var_importance_pct,
        main = "Variable Importance (% of Splits)",
        ylab = "% of Total Splits",
        col  = "steelblue",
        las  = 2,
        cex.names = 0.8)

# ========== RMSE Comparison ========== #

rmse_bart  <- sqrt(mean((y_bart - y_pred_bart)^2))
rmse_linear <- sqrt(mean((y - fitted(log_ols))^2))

cat("\nRMSE Comparison:\n")
cat("OLS (log price):       ", round(rmse_linear, 4), "\n")
cat("BART (best base =", best_base, "):", round(rmse_bart, 4), "\n")


# ==============================================
#            Partial Dependence Plots 
# ==============================================

pdp_plot <- function(bart_model, X_mat, var_idx, var_name,
                     n_grid = 20, n_sample = 500) {

  # subsample rows for speed
  sample_idx <- sample(nrow(X_mat), n_sample)
  X_sub <- X_mat[sample_idx, ]

  grid_vals <- seq(min(X_mat[, var_idx]),
                   max(X_mat[, var_idx]),
                   length.out = n_grid)

  pd_means <- numeric(n_grid)
  pd_lower <- numeric(n_grid)
  pd_upper <- numeric(n_grid)

  for (g in seq_along(grid_vals)) {
    X_temp <- X_sub
    X_temp[, var_idx] <- grid_vals[g]

    pred <- predict(bart_model, newdata = X_temp)
    draw_means <- rowMeans(pred)

    pd_means[g] <- mean(draw_means)
    pd_lower[g] <- quantile(draw_means, 0.025)
    pd_upper[g] <- quantile(draw_means, 0.975)
  }

  plot(grid_vals, pd_means, type = "l", lwd = 2, col = "blue",
       main = paste("Partial Dependence:", var_name),
       xlab = var_name, ylab = "Predicted log(Price)",
       ylim = range(c(pd_lower, pd_upper)))
  polygon(c(grid_vals, rev(grid_vals)),
          c(pd_upper, rev(pd_lower)),
          col = rgb(0, 0, 1, 0.15), border = NA)
}

par(mfrow = c(2, 3))
top_vars <- order(colMeans(bart_final$varcount), decreasing = TRUE)[1:6]

for (v in top_vars) {
  pdp_plot(bart_final, X_bart,
           var_idx  = v,
           var_name = colnames(X_bart)[v])
}
par(mfrow = c(1, 1))