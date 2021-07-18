import numpy as np
import pandas as pd
import xgboost as xgboost
import catboost as catboost
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from abc import ABC, abstractmethod

class Tuner(ABC):

  """ 
  Hyperparameter tuner for gradient boosted models via Bayesian optimization

  Supported Libraries
  -------------------
  - XGBoost
  - Catboost
  
  Input
  -----
  X : pd.DataFrame
    Input data with shape (n_samples, n_features)

  y : [list, np.array, pd.Series]
    Target data with shape (n_samples,)

  cats : list
    Categorical features

  model_config : dict
    Constant parameters for model (lossfn, learning_rate, etc)

  cv_config : dict
    Cross validation settings (n_folds, stratified, verbose_eval, etc)

  pbounds : dict
    Domains for hyperparameter search

  Examples
  --------
  >>> tuner = XGB_Tuner(X, y, cats, model_config, cv_config, pbounds)
  >>> tuner.optimize(n_iter)
  >>> tuner.model.predict(xgb.DMatrix(test))
  
  Sources
  -------
  - https://github.com/fmfn/BayesianOptimization
  - https://xgboost.readthedocs.io/en/latest/
  - https://catboost.ai/
  """

  LIBRARY = None

  # Tunable integer hyperparameters
  INTEGER_PARAMS = [
    'n_estimators', 'num_trees', 'num_boost_round', 
    'best_model_min_trees', 'depth', 'max_depth', 
    'min_data_in_leaf', 'min_child_samples', 'max_leaves', 
    'num_leaves', 'one_hot_max_size']

  def __init__(self, X, y, cats, model_config, cv_config, pbounds):
    if type(X) != pd.DataFrame:
      raise TypeError('Input data X must be a pandas DataFrame')
    self.X, self.y, self.cats = X, y, cats
    self.model_config = model_config
    self.cv_config = cv_config
    self.pbounds = pbounds

  @abstractmethod
  def format_data(self):
    """ catboost.Pool() | xgboost.DMatrix() """

  def integerize(self, params):
    # Rounding int parameters to nearest integer
    for p in set(params.keys()).intersection(self.INTEGER_PARAMS):
      params[p] = int(round(params[p]))
    return params

  def cross_val(self, params):
    # Wrapper for XGBoost and CatBoost cross validation
    return self.LIBRARY.cv(self.train_data, self.integerize(params), **self.cv_config)

  def objective(self, **params):
    # K-Fold CV for arbitrary loss function
    cv_results = self.cross_val({**params, **self.model_config})
    loss_name = self.model_config['objective']
    loss_value = cv_results[f'test-{loss_name}-mean'].iloc[-1]
    return -1*loss_value

  def optimize(self, n_iter, init_points, opt_config, transformer = None):
    # Initializing optimizer with objective and hyperparamater ranges then training
    print(f'Warming Up Sampler...\nOptimizing for {n_iter+init_points} iterations...\n\n')
    self.opt = BayesianOptimization(self.objective, self.pbounds, bounds_transformer = transformer)
    self.opt.maximize(n_iter = n_iter, init_points = init_points, **opt_config)
    self.train()

    # Storing (hyperparameter(s), loss) data in DataFrame
    hyper_names = self.opt.space.keys
    opt_results = np.hstack((self.opt.space.params, self.opt.space.target.reshape(-1,1)))
    self.opt_results = pd.DataFrame(opt_results, columns = hyper_names + ['target'])

  def train(self):
    # Training with optimal hyperparameters from K-Fold cross validation
    best_params = {**self.opt.max['params'], **self.model_config}
    self.model = self.LIBRARY.train(self.train_data, self.integerize(best_params))

  def predict(self, data):
    return self.model.predict(data)

class XGB_Tuner(Tuner):

  """ XGBoost Bayesian model selector """
  LIBRARY = xgboost

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.train_data = self.format_data(self.X, self.y)

  def format_data(self, X, y = None):
    return xgboost.DMatrix(data = X, label = y)

class CB_Tuner(Tuner):

  """ CatBoost Bayesian model selector """
  LIBRARY = catboost

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.train_data = self.format_data(self.X, self.y, self.cats)

  def format_data(self, X, y = None, cats = None):
    return catboost.Pool(data = X, label = y, cat_features = cats)