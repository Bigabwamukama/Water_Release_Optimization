import json
import numpy as np
import constants as cts
import pandas as pd

def extract_water_data_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        water_levels = df["LAKE_LEVEL_(m)"].dropna().tolist()
        discharges = df["Total discharge KPS& NPS"].dropna().tolist()
        return {
            "waterlevel": water_levels,
            "discharge": discharges
        }       
    except Exception as e:
        print(f"Error reading file: {e}")
        return None



def split_and_save_data(data, train_ratio = 0.8, 
                        train_filename = cts.TRAIN_DATA_FILE_NAME, 
                        test_filename = cts.TEST_DATA_FILE_NAME, seed=None):
    if seed is not None:
        np.random.seed(seed)  
    num_samples = len(data["waterlevel"])
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * train_ratio)
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    train_data = {key: np.array(data[key])[train_indices].tolist() for key in data}
    test_data = {key: np.array(data[key])[test_indices].tolist() for key in data}
    print(f"Train set length: {int(len(train_data['waterlevel']))}")
    print(f"Test set length : {int(len(test_data['waterlevel']))}")
    with open(train_filename, "w") as f_train:
        json.dump(train_data, f_train, indent=2)
    with open(test_filename, "w") as f_test:
        json.dump(test_data, f_test, indent=2)

def load_data(train_filename=cts.TRAIN_DATA_FILE_NAME, 
              test_filename = cts.TEST_DATA_FILE_NAME ):
    """Loads train and test sets from JSON files and returns them as dictionaries."""
    with open(train_filename, "r") as f_train:
        train_data = json.load(f_train)
    with open(test_filename, "r") as f_test:
        test_data = json.load(f_test)
    return [train_data, test_data]


if __name__ == "__main__":
    excel_path = '../sheets/Compiled.xlsx'
    dataset = extract_water_data_from_excel(excel_path)
    print("Splitting data into train and test set")
    split_and_save_data(dataset,train_ratio=0.9,seed=None)
    print(f"Done")