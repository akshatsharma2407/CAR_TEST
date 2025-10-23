import os
import logging
import joblib
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

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

def load_data(
    train_path: pd.DataFrame, test_path: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug('data loaded')
        return train_data, test_data
    except FileNotFoundError:
        logger.error(f'File not found in either of given locations {__file__} -> {train_data}, {test_data}')
        raise
    except Exception as e:
        logger.error(f'unexpected error in {__file__} -> load_data')
        raise


def transformations(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, TransformerMixin]:
    try:
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        scaler.fit(train_data)
        train_processed_data = scaler.transform(train_data)
        test_processed_data = scaler.transform(test_data)
        logger.debug('transformation done on data')
        return train_processed_data, test_processed_data, scaler
    except Exception as e:
        logger.error(f'found unexpected error in {__file__} -> transformation function')
        raise


def save_artifacts(
    data_path: str,
    scaler_path: str,
    scaler: TransformerMixin,
    train_processed_data: pd.DataFrame,
    test_processed_data: pd.DataFrame,
) -> None:
    try:
        data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"))
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"))
        logger.debug('artifacts saved')
    except ModuleNotFoundError:
        logger.error(f'Either of module path is not found {__file__} -> {data_path}, {scaler}')
        raise
    except Exception as e:
        logger.error(f'Found unexpected error {__file__} -> save_artifacts')
        raise


def main() -> None:
    try:
        train_data, test_data = load_data(
            train_path="data/raw/train.csv", test_path="data/raw/test.csv"
        )
        train_processed_data, test_processed_data, scaler = transformations(
            train_data, test_data
        )
        save_artifacts(
            data_path="data/processed",
            scaler_path="models/scaler.joblib",
            scaler=scaler,
            train_processed_data=train_processed_data,
            test_processed_data=test_processed_data,
        )
        logger.debug('main function executed')
    except Exception as e:
        logger.error(f'Found unexpected error in {__file__} -> main')
        raise

if __name__ == "__main__":
    main()
