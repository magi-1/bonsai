# Bonsai: Gradient Boosted Trees + Bayesian Optimization

Bonsai is a wrapper for the XGBoost and Catboost model training pipelines that leverages Bayesian optimization for computationally efficient hyperparameter tuning.

Despite being a very small package, it has access to nearly all of the configurable parameters in XGBoost and CatBoost as well as the BayesianOptimization package allowing users to specify unique objectives, metrics, parameter search ranges, and search policies. This is made possible thanks to the strong similaries between both libraries.

```console
$ pip install bonsai-tree
```

References/Dependencies:
- [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
- [CatBoost](https://catboost.ai/docs/concepts/python-reference_parameters-list.html)
- [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)

## Why use Bonsai?

Grid search and random search are the most commonly used algorithms for exploring the hyperparameter space for a wide range of machine learning models. While effective for optimizing over low dimensional hyperparameter spaces (ex: few regularization terms), these methods do not scale well to models with a large number of hyperparameters such as gradient boosted trees.

Bayesian optimization on the other hand *dynamically* samples from the hyperparameter space with the goal of minimizing uncertaintly about the underlying objective function. For the case of model optimization, this consists of *iteratively* building a prior distribution of functions over the hyperparameter space and sampling with the goal of minimizing the posterior variance of the loss surface (via Gaussian Processes).

### Model Configuration

Since Bonsai is simply a wrapper for both XGBoost and CatBoost, the model_params dict is synonymous with the params argument for both [catboost.fit()](https://catboost.ai/docs/concepts/python-reference_parameters-list.html) and [xgboost.fit()](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training). Additionally, you must encode your categorical features as usual depending on which library you are using (XGB: One-Hot, CB: Label). 

Below is a simple example of binary classification using CatBoost:

``` python
# label encoded training data
X = train.drop(target, axis = 1)
y = train[target]

# same args as catboost.train(...)
model_params = dict(objective = 'Logloss', verbose = False)

# same args as catboost.cv(...)
cv_params = dict(nfold = 5)
```

The pbounds dict as seen below specifies the hyperparameter bounds over which the optimizer will search. Additionally, the opt_config dictionary is for configuring the optimizer itself. Refer to the [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) documentation to learn more.  

``` python
# defining parameter search ranges
pbounds = dict(
  eta = (0.15, 0.4), 
  n_estimators = (200,2000), 
  max_depth = (4, 8)
)

# 10 warm up samples + 10 optimizing steps
n_iter, init_points= 10, 10

# to learn more about customizing your search policy:
# BayesianOptimization/examples/exploitation_vs_exploration.ipynb
opt_config = dict(acq = 'ei', xi = 1e-2)
```

### Tuning and Prediction

All that is left is to initialize and optimize. 

``` python
from bonsai.tune import CB_Tuner

# note that 'cats' is a list of categorical feature names
tuner = CB_Tuner(X, y, cats, model_params, cv_params, pbounds)
tuner.optimize(n_iter, init_points, opt_config, bounds_transformer)
``` 

After the optimal parameters are found, the model is trained and stored internally giving full access to the CatBoost model. 

``` python
test_pool = catboost.Pool(test, cat_features = cats)
preds = tuner.model.predict(test_pool, prediction_type = 'Probability')
```

Bonsai also comes with a parallel coordinates plotting functionality allowing users to further narrow down their parameter search ranges as needed.

``` python
from bonsai.utils import parallel_coordinates

# DataFrame with hyperparams and observed loss
results = tuner.opt_results
parallel_coordinates(results)
```

<p align="center">
  <img src="https://github.com/magi-1/bonsai/blob/8658ed04ce53040f52caed86680aa8d3f6a9354c/images/param_plot.png" />
</p>
