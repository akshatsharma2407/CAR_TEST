import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('reports/errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_params(param_path: str) -> float:
    try:
        test_size = yaml.safe_load(open(param_path, "r"))["data_ingestion"]["test_size"]
        logger.debug('test size retrieved')
        return test_size
    except FileNotFoundError:
        logger.error(f"File do not exist at the given location {__file__} -> {param_path}")
        raise
    except Exception as e:
        logger.error(f"Got an unexpected error in {__file__} -> load_params")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = (
            pd.read_csv(data_path)
            .drop(columns=["Image_List"])
            .select_dtypes(include=[int, float, bool])
        )
        logger.debug('data loaded successfully')
        return df
    except FileNotFoundError as e:
        logger.error((f"\n\nFile do not exist at the given location {__file__} -> {data_path}, {e}\n\n"))
        raise
    except Exception as e:
        logger.error(f"Got unexpected error in {__file__} -> load_data")
        raise

def save_data(
    data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> None:
    try:
        data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"))
        test_data.to_csv(os.path.join(data_path, "test.csv"))
        logger.debug('data saved')
    except Exception as e:
        logger.error(f"Got unexpected error in {__file__} -> save data")
        raise

def main() -> None:
    try:
        test_size = load_params("params.yaml")
        df = load_data(
            "C:/Users/aksha/OneDrive/Desktop/cars_mlops_practice/sample_data.csv"
        )
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data("data/raw", train_data, test_data)
        logger.debug('main function executed')
    except Exception as e:
        logger.error(f"got unexpected error in data_ingestion {__file__} -> main function")
        raise


if __name__ == "__main__":
    main()
