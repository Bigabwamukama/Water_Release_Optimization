import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("Loading tensorflow, wait patiently......")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
import constants as cts
import dataset_handler as dh

class LossLogger(keras.callbacks.Callback):
    def __init__(self, log_file, model_save_dir=cts.NN_TEMP_WEIGHTS_DIR):
        super().__init__()
        self.log_file = log_file
        self.model_save_dir = model_save_dir
        self.epochs = []
        self.train_losses = []
        os.makedirs(model_save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.train_losses.append(logs['loss'])
        df = pd.DataFrame({
            'Epoch': self.epochs,
            'Train Loss': self.train_losses
        })
        df.to_csv(self.log_file, index=False)
        model_filename = os.path.join(self.model_save_dir, f"model_epoch_{epoch + 1}.keras")
        self.model.save(model_filename)


def save_scaler(scaler, filename=cts.NN_SCALERSX):
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(scaler_params, f)

def load_scaler(filename=cts.NN_SCALERSX):
    with open(filename, 'r') as f:
        scaler_params = json.load(f)    
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])   
    return scaler

def compute_nse(observed, predicted):
    return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

def build_nn_model(train_set, model_path=cts.NN_WEIGHTS, 
                   scaler_pathx=cts.NN_SCALERSX,scaler_pathy=cts.NN_SCALERSY, 
                   log_file=cts.NN_LOGFILE, weights_folder=cts.NN_TEMP_WEIGHTS_DIR):
    X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
    y_train = np.array(train_set["discharge"]).reshape(-1, 1) 
    num_samples = len(X_train)  
    batch_size = int(np.sqrt(num_samples))
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()   
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)  
    save_scaler(scaler_X, scaler_pathx)   
    save_scaler(scaler_y, scaler_pathy) 
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')   
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_folder, "nn_epoch_{epoch}.keras"),
        save_weights_only=True,
        save_freq='epoch',
        verbose=0
    )
    loss_logger = LossLogger(log_file=log_file)   
    model.fit(
        X_train_scaled, y_train_scaled,
        epochs=1000,
        batch_size= batch_size,
        callbacks=[early_stopping, checkpoint_callback, loss_logger],
        verbose=0
    )   
    model.save(model_path)    
    return model, scaler_X, scaler_y

def evaluate_nn_model(model, train_set, test_set, scaler_X, scaler_y):
    def evaluate(X, y, dataset_name):
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)
        y_pred_scaled = model.predict(X_scaled,verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_orig = scaler_y.inverse_transform(y_scaled)
        mse = mean_squared_error(y_orig, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_orig, y_pred)
        correlation_matrix = np.corrcoef(y_orig.flatten(), y_pred.flatten())
        pearson_r = correlation_matrix[0, 1]
        r2 = pearson_r ** 2
        nse = compute_nse(y_orig.flatten(), y_pred.flatten())
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nse': nse,
            'corr': pearson_r
        }
    X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
    y_train = np.array(train_set["discharge"]).reshape(-1, 1)
    X_test = np.array(test_set["waterlevel"]).reshape(-1, 1)
    y_test = np.array(test_set["discharge"]).reshape(-1, 1)
    train_metrics = evaluate(X_train, y_train, "train")
    test_metrics = evaluate(X_test, y_test, "test")
    i = 0
    figure_save_path = [cts.NN_TRAIN_SET_EVALUATION_PLOT,cts.NN_TEST_SET_EVALUATION_PLOT]
    for dataset_name, X, y, metrics in [("Train", X_train, y_train, train_metrics), 
                                         ("Test", X_test, y_test, test_metrics)]:
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)
        y_pred_scaled = model.predict(X_scaled,verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_orig = scaler_y.inverse_transform(y_scaled)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_orig, y_pred, alpha=0.7, label=f"{dataset_name} Data")
        plt.plot([min(y_orig), max(y_orig)], [min(y_orig), max(y_orig)], 'r--')
        plt.xlabel('Actual Discharge')
        plt.ylabel('Predicted Discharge')
        plt.title(f'Neural Network: Actual vs. Predicted Discharge ({dataset_name})')
        plt.text(
        0.05, 0.85, f"$R^2$ = {test_metrics['r2']:.4f}\n",
        transform=plt.gca().transAxes, fontsize=12, color="black",
        bbox=dict(facecolor="white", alpha=0.8)
        )
        plt.legend(loc='upper right')
        plt.savefig(figure_save_path[i], bbox_inches='tight')
        i += 1
        plt.show()

    print("======Performance Metrics on Test set======")
    dump_evaluation_metrics(test_metrics)
    return {"train": train_metrics, "test": test_metrics}

def predict_discharge_nn(model_path=cts.NN_WEIGHTS, scaler_pathx=cts.NN_SCALERSX,scaler_pathy=cts.NN_SCALERSY, water_levels = None, return_as_array=False):
    if water_levels is None:
        return None
    model = keras.models.load_model(model_path)
    scaler_X = load_scaler(scaler_pathx)
    scaler_Y = load_scaler(scaler_pathy)
    if isinstance(water_levels, (int, float)):
        water_levels = [water_levels]   
    if isinstance(water_levels, np.ndarray):
        water_levels = water_levels.tolist()
    X_scaled = scaler_X.transform(np.array(water_levels).reshape(-1, 1))
    discharge_scaled = model.predict(X_scaled,verbose=0)
    discharge = scaler_Y.inverse_transform(discharge_scaled)
    discharge = discharge.flatten()   
    if return_as_array:
        return discharge
    else:
        return discharge.tolist()
    
def plot_loss(csv_file_path=cts.NN_LOGFILE):
    df = pd.read_csv(csv_file_path)
    if 'Epoch' not in df.columns or 'Train Loss' not in df.columns:
        print("CSV file must contain 'Epoch', 'Train Loss' columns.")
        return
    epochs = df['Epoch']
    train_losses = df['Train Loss']
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Variation of Train  Loss with Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def dump_evaluation_metrics(results):
    print(f" MAE  : {results['mae']}")
    print(f" RMSE : {results['rmse']}")
    print(f" NSE  : {results['nse']}")
    print(f" CORR : {results['corr']}")
    print(f" R2   : {results['r2']}")

if __name__ == "__main__":
    [traindata,testdata] = dh.load_data()
    model, scaler_X, scaler_y= build_nn_model(train_set=traindata)
    metrics = evaluate_nn_model(model=model,train_set=traindata,test_set=testdata,scaler_X=scaler_X,scaler_y=scaler_y)
    plot_loss()
    