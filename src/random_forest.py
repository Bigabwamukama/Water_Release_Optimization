import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import constants as cts
import dataset_handler as dh

def nash_sutcliffe_efficiency(observed, predicted):
    observed_mean = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - observed_mean) ** 2)   
    if denominator == 0:
        return np.nan       
    return 1 - (numerator / denominator)

def build_forest_model(train_set, model_path=cts.RAND_WEIGHTS):
    X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
    y_train = np.array(train_set["discharge"])
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                               param_grid, cv=5, n_jobs=-1, 
                               scoring='neg_mean_squared_error')
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_path)
    return best_model

def compute_nse(y_true, y_pred):
    return (1 - np.sum((y_true - y_pred) ** 2) /
            np.sum((y_true - np.mean(y_true)) ** 2))

def evaluate_forest_model(model, train_set, test_set):
    X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
    y_train = np.array(train_set["discharge"])
    X_test = np.array(test_set["waterlevel"]).reshape(-1, 1)
    y_test = np.array(test_set["discharge"])
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    sorted_indices = np.argsort(X_train.flatten())
    sorted_x = X_train[sorted_indices]
    sorted_y_pred = y_train_pred[sorted_indices]
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, label="Training Data", alpha=0.6)
    plt.plot(sorted_x, sorted_y_pred, color="red", linewidth=2, label="Predicted")
    plt.xlabel("Water Level")
    plt.ylabel("Discharge")
    plt.grid()
    plt.legend(loc="lower right")
    plt.title("Random Forest Regression: Training Data")
    plt.savefig(cts.RAND_TRAIN_SET_EVALUATION_PLOT, bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color="blue", alpha=0.6, label="Test Data")
    plt.plot(y_test, y_test, color="red", linestyle="--", linewidth=2, 
             label="Ideal Fit (y = x)")
    plt.xlabel("Observed Discharge")
    plt.ylabel("Predicted Discharge")
    plt.grid()
    plt.legend(loc="lower right")
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    corr, _ = pearsonr(y_test, y_test_pred)
    r2 = corr ** 2  
    nse = compute_nse(y_test, y_test_pred)
    plt.text(
        0.05, 0.85, f"$R^2$ = {r2:.4f}\n",
        transform=plt.gca().transAxes, fontsize=12, color="black",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    plt.title("Predicted vs Observed Discharge (Test Set)")
    plt.savefig(cts.RAND_TEST_SET_EVALUATION_PLOT, bbox_inches='tight')
    plt.show()
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'nse': nse,
        'corr': corr
    }
    print("======Performance Metrics on Test set======")
    dump_evaluation_metrics(metrics)
    return metrics

def predict_discharge_forest(model_path=cts.RAND_WEIGHTS, water_levels=None, 
                             return_as_array=False):
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
    model =  build_forest_model(train_set=train_set)
    metrics = evaluate_forest_model(model=model,train_set=train_set,
                                    test_set=test_set)
