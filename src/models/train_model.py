import joblib
import numpy as np
import pandas as pd
import yaml
import logging
import os
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import mlflow

mlflow.set_tracking_uri('http://ec2-13-203-154-81.ap-south-1.compute.amazonaws.com:5000/')

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

file_handler = logging.FileHandler('reports/errors.log')
file_handler.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_params(param_path: str) -> int:
    try:
        n_estimator = yaml.safe_load(open(param_path, "r"))["train_model"]["n_estimators"]
        logger.debug('params loaded')
        return n_estimator
    except FileNotFoundError:
        logger.error(f'File not found at given location {__file__} -> {param_path}')
    except Exception as e:
        logger.error(f'Found unexpected error in {__file__}')
        raise


def load_data(data_path: str) -> tuple[np.array, np.array]:
    try:
        train_data = pd.read_csv(data_path)
        xtrain = train_data.drop(columns="Price").values
        ytrain = train_data["Price"].values
        logger.debug('data loaded')
        return xtrain, ytrain
    except ModuleNotFoundError:
        logger.error(f'Module not found at given location {__file__} -> {data_path}')
        raise
    except Exception as e:
        logger.error(f'Found unexpected error in {__file__} -> load_data')
        raise


def train_model(n_estimator: int, xtrain: np.array, ytrain: np.array) -> BaseEstimator:
    try:
        model = GradientBoostingRegressor(n_estimators=n_estimator)

        param_grid = {
            "n_estimators": [100, 50],
            "max_depth": [3, 4]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)

        grid_search.fit(xtrain,ytrain)
            
        logger.debug('model trained')
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f'Found unexpected error in {__file__} -> train_model')
        raise


def save_model(model_path: str, model: BaseEstimator) -> None:
    try:
        joblib.dump(model, model_path)
        logger.debug('ml model saved')
    except Exception as e:
        logger.error(f'Found unexpected error in {__file__} -> save_model')
        raise


def main() -> None:
    try:
        mlflow.sklearn.autolog()
        mlflow.set_experiment('hyperparams tuning')
        with mlflow.start_run():
            n_estimator = load_params(param_path="params.yaml")
            xtrain, ytrain = load_data(data_path="data/processed/train_processed.csv")
            model = train_model(n_estimator=n_estimator, xtrain=xtrain, ytrain=ytrain)
            save_model(model_path="models/model.pkl", model=model)
            logger.debug('main function executed')
    except Exception as e:
        logger.error(f'Found Unexpected error in {__file__} -> main')
        raise


if __name__ == "__main__":
    main()
