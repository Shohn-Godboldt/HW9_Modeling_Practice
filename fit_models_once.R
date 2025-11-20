## fit_models_once.R
## Run this ONCE to fit all models and save them to saved_models.RData

library(tidyverse)
library(tidymodels)
library(janitor)
library(rpart.plot)
library(vip)
library(baguette)
library(glmnet)
library(ranger)

set.seed(123)

#--------------------------------------------------
# 1. Read and clean data
#--------------------------------------------------

bike_raw <- read_csv(
  "SeoulBikeData.csv",
  locale = locale(encoding = "latin1"),
  show_col_types = FALSE
)

bike <- bike_raw %>%
  clean_names() %>%
  mutate(
    holiday         = factor(holiday),
    seasons         = factor(seasons),
    functioning_day = factor(functioning_day)
  )

# Drop date column if present
if ("date" %in% names(bike)) {
  bike <- bike %>% select(-date)
}

# Train / test split
set.seed(123)
bike_split <- initial_split(bike, prop = 0.8, strata = rented_bike_count)
bike_train <- training(bike_split)
bike_test  <- testing(bike_split)

# 5-fold CV (faster than 10-fold)
set.seed(123)
bike_folds <- vfold_cv(bike_train, v = 5, strata = rented_bike_count)

#--------------------------------------------------
# 2. Recipe (extra safeguards for glmnet)
#--------------------------------------------------

bike_rec <- recipe(rented_bike_count ~ ., data = bike_train) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%   # full one-hot
  step_zv(all_predictors()) %>%                              # remove zero var
  step_corr(all_predictors(), threshold = 0.99) %>%          # remove collinear
  step_nzv(all_predictors()) %>%                             # remove near-zero var
  step_normalize(all_numeric_predictors())

# Prep once to confirm it works
prep(bike_rec)

#--------------------------------------------------
# 3. Multiple Linear Regression (MLR)
#--------------------------------------------------

cat("Fitting MLR...\n")

mlr_spec <- linear_reg() %>%
  set_engine("lm")

mlr_wf <- workflow() %>%
  add_model(mlr_spec) %>%
  add_recipe(bike_rec)

mlr_fit <- mlr_wf %>%
  fit(data = bike_train)

#--------------------------------------------------
# 4. LASSO (smaller grid to be faster)
#--------------------------------------------------

cat("Tuning LASSO...\n")

lasso_spec <- linear_reg(
  penalty = tune(),
  mixture = 1    # pure LASSO
) %>%
  set_engine("glmnet")

lasso_wf <- workflow() %>%
  add_model(lasso_spec) %>%
  add_recipe(bike_rec)

lasso_grid <- grid_regular(
  penalty(range = c(-4, 0)),   # 10^-4 to 10^0
  levels = 10
)

lasso_res <- lasso_wf %>%
  tune_grid(
    resamples = bike_folds,
    grid      = lasso_grid,
    metrics   = metric_set(rmse, mae)
  )

# (optional) look at metrics if you want
# collect_metrics(lasso_res)

best_lasso <- tune::select_best(lasso_res, metric = "rmse")

final_lasso_wf <- lasso_wf %>%
  finalize_workflow(best_lasso)

final_lasso_fit <- final_lasso_wf %>%
  fit(data = bike_train)

#--------------------------------------------------
# 5. Regression tree (smaller grid)
#--------------------------------------------------

cat("Tuning regression tree...\n")

tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_wf <- workflow() %>%
  add_model(tree_spec) %>%
  add_recipe(bike_rec)

tree_grid <- grid_regular(
  cost_complexity(range = c(-4, -2)),
  tree_depth(range = c(2, 8)),
  min_n(range = c(5, 25)),
  levels = 3
)

tree_res <- tree_wf %>%
  tune_grid(
    resamples = bike_folds,
    grid      = tree_grid,
    metrics   = metric_set(rmse, mae)
  )

best_tree <- tune::select_best(tree_res, metric = "rmse")

final_tree_wf <- tree_wf %>%
  finalize_workflow(best_tree)

final_tree_fit <- final_tree_wf %>%
  fit(data = bike_train)

#--------------------------------------------------
# 6. Bagged tree (smaller grid)
#--------------------------------------------------

cat("Tuning bagged tree...\n")

bag_spec <- bag_tree(
  tree_depth = tune(),
  min_n      = tune()
) %>%
  set_engine("rpart", times = 30) %>%   # fewer trees to speed up
  set_mode("regression")

bag_wf <- workflow() %>%
  add_model(bag_spec) %>%
  add_recipe(bike_rec)

bag_grid <- grid_regular(
  tree_depth(range = c(2, 8)),
  min_n(range = c(5, 25)),
  levels = 3
)

bag_res <- bag_wf %>%
  tune_grid(
    resamples = bike_folds,
    grid      = bag_grid,
    metrics   = metric_set(rmse, mae)
  )

best_bag <- tune::select_best(bag_res, metric = "rmse")

final_bag_wf <- bag_wf %>%
  finalize_workflow(best_bag)

final_bag_fit <- final_bag_wf %>%
  fit(data = bike_train)

#--------------------------------------------------
# 7. Random forest (smaller grid, fewer trees)
#--------------------------------------------------

cat("Tuning random forest...\n")

rf_spec <- rand_forest(
  mtry  = tune(),
  trees = 300,
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(bike_rec)

rf_grid <- grid_regular(
  mtry(range = c(3, 12)),
  min_n(range = c(5, 25)),
  levels = 3
)

rf_res <- rf_wf %>%
  tune_grid(
    resamples = bike_folds,
    grid      = rf_grid,
    metrics   = metric_set(rmse, mae)
  )

best_rf <- tune::select_best(rf_res, metric = "rmse")

final_rf_wf <- rf_wf %>%
  finalize_workflow(best_rf)

final_rf_fit <- final_rf_wf %>%
  fit(data = bike_train)

#--------------------------------------------------
# 8. Save everything for use in the QMD
#--------------------------------------------------

cat("Saving fitted models to saved_models.RData...\n")

save(
  bike, bike_train, bike_test, bike_split, bike_folds, bike_rec,
  mlr_fit, mlr_wf,
  final_lasso_fit, final_lasso_wf,
  final_tree_fit, final_tree_wf,
  final_bag_fit, final_bag_wf,
  final_rf_fit, final_rf_wf,
  file = "saved_models.RData"
)

cat("Done! Models fitted and saved.\n")
