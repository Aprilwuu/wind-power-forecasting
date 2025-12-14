import lightgbm as lgb
from typing import Optional

class LGBMModel:
    def __init__(self, params, num_boost_round = 500):
        self.params = params
        self.num_boost_round = num_boost_round
        self.model: Optional[lgb.Booster] = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        train_data = lgb.Dataset(X_train, label=y_train)

        # Define callbacks manually for full version compatibility
        callbacks = []

        # Add early stopping if supported
        if hasattr(lgb, "early_stopping"):
            callbacks.append(lgb.early_stopping(stopping_rounds=50))

        # Add logging control callback (replaces verbose_eval)
        if hasattr(lgb, "log_evaluation"):
            callbacks.append(lgb.log_evaluation(period=100))

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )
        else:
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.num_boost_round,
                callbacks=callbacks,
            )

    def predict(self, X):
            if self.model is None:
                 raise ValueError("Model is not trained yet.")
            best_iter = self.model.best_iteration or self.model.current_iteration()
            return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def save(self, path):
        if self.model is None:
             raise ValueError("Models is not trained yet.")
        self.model.save_model(path)

def create_lgbm_model() -> LGBMModel:
     """
     Factory used by train/loop.py
     """
     params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leave": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_threads": -1,
     }
     return LGBMModel(params=params, num_boost_round=500)