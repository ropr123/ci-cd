import os
from sklearn import datasets
import pandas as pd

def load_and_save_dataset(dataset_name: str = "iris", output_dir: str = "data"):
    """Load a dataset from scikit-learn and save it as a CSV file.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset to load. Currently supports "iris" and "digits".
    output_dir: str
        Directory (relative to this script) where the CSV will be saved.
    """
    # Load the requested dataset
    if dataset_name == "iris":
        data = datasets.load_iris(as_frame=True)
    elif dataset_name == "digits":
        data = datasets.load_digits(as_frame=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    data.frame.to_csv(csv_path, index=False)
    print(f"{dataset_name} dataset saved to {csv_path}")

if __name__ == "__main__":
    # Example usage: load the iris dataset and save it under the "data" folder
    load_and_save_dataset()
