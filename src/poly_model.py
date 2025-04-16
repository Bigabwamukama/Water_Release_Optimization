import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import constants as cts
import dataset_handler as dh


def build_poly_model(train_set, model_path=cts.POLY_WEIGHTS, transformer_path=cts.POLY_TRANSFORMER):
    X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
    y_train = np.array(train_set["discharge"]).reshape(-1, 1)
    poly_transformer = PolynomialFeatures(degree=3)
    model = make_pipeline(poly_transformer, LinearRegression())
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    joblib.dump(poly_transformer, transformer_path)
    return model, poly_transformer

def load_poly_model(model_path=cts.POLY_WEIGHTS, transformer_path=cts.POLY_TRANSFORMER):
    model = joblib.load(model_path)
    poly_transformer = joblib.load(transformer_path)
    return model, poly_transformer

def compute_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def evaluate_poly_model(model, train_set, test_set):
    X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
    y_train = np.array(train_set["discharge"]).reshape(-1, 1)
    X_test = np.array(test_set["waterlevel"]).reshape(-1, 1)
    y_test = np.array(test_set["discharge"]).reshape(-1, 1)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    lin_reg = model.named_steps["linearregression"]
    coeffs = lin_reg.coef_[0]
    intercept = lin_reg.intercept_[0]
    equation = f"y = {intercept:.2f} + {coeffs[1]:.2f}x + {coeffs[2]:.2f}x² + {coeffs[3]:.2f}x³"
    sorted_indices = np.argsort(X_train.flatten())
    sorted_x = X_train[sorted_indices]
    sorted_y_pred = y_train_pred[sorted_indices]
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, label="Training Data", alpha=0.6)
    plt.plot(sorted_x, sorted_y_pred, color="red", linewidth=2, label="Best Fit Curve")
    plt.xlabel("Water Level")
    plt.ylabel("Discharge")
    plt.grid()
    plt.legend(loc="lower right")
    plt.text(
        0.05, 0.85, equation,
        transform=plt.gca().transAxes, fontsize=12, color="black",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    plt.title("Polynomial Regression: Training Data")
    plt.savefig(cts.POLY_TRAIN_SET_EVALUATION_PLOT, bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color="blue", alpha=0.6, label="Test Data")
    plt.plot(y_test, y_test, color="red", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
    plt.xlabel("Observed Discharge")
    plt.ylabel("Predicted Discharge")
    plt.grid()
    plt.legend(loc="lower right")
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    correlation_matrix = np.corrcoef(y_test.flatten(), y_test_pred.flatten())
    pearson_r = correlation_matrix[0, 1]
    r2 = pearson_r ** 2
    nse = compute_nse(y_test.flatten(), y_test_pred.flatten())
    plt.text(
        0.05, 0.85, f"$R^2$ = {r2:.4f}\n",
        transform=plt.gca().transAxes, fontsize=12, color="black",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    plt.title("Predicted vs Observed Discharge (Test Set)")
    plt.savefig(cts.POLY_TEST_SET_EVALUATION_PLOT, bbox_inches='tight')
    plt.show()
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'nse': nse,
        'corr': pearson_r
    }
    print("======Performance Metrics on Test set======")
    dump_evaluation_metrics(metrics)
    return metrics

def predict_discharge_poly(model_path=cts.POLY_WEIGHTS, water_levels=None, return_as_array=False):
    if water_levels is None:
        return None
    model = joblib.load(model_path)
    if isinstance(water_levels, (int, float)):
        water_levels = [water_levels]  
    if isinstance(water_levels, np.ndarray):
        water_levels = water_levels.tolist()
    X = np.array(water_levels).reshape(-1, 1)
    discharge = model.predict(X).flatten()  
    if return_as_array:
        return discharge 
    else:
        return discharge.tolist()

def dump_evaluation_metrics(results):
    print(f" MAE  : {results['mae']}")
    print(f" RMSE : {results['rmse']}")
    print(f" NSE  : {results['nse']}")
    print(f" CORR : {results['corr']}")
    print(f" R2   : {results['r2']}")

if __name__ == "__main__":
    [train_set,test_set] = dh.load_data()
    model, poly_transformer = build_poly_model(train_set=train_set)
    metrics = evaluate_poly_model(model=model,train_set=train_set,test_set=test_set)
    
    




