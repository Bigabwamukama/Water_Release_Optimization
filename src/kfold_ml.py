import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataset_handler as dh
import constants as cts
import joblib
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import  GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
print("Loading tensorflow, wait patiently......")
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 
from tensorflow import keras
print("Done Loading tensorflow")
plt.rcParams.update({'font.size': 13})

def save_scaler_fnn(scaler, filename=cts.NN_SCALERSX):
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(scaler_params, f)

def load_scaler_fnn(filename=cts.NN_SCALERSX):
    with open(filename, 'r') as f:
        scaler_params = json.load(f)   
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])   
    return scaler

def compute_nse(observed, predicted):
    return (1 - (np.sum((observed - predicted) ** 2) / 
                 np.sum((observed - np.mean(observed)) ** 2)))

def build_feedforward_nn(kf, X, y, epochs=1000, batch_size=300):
    fold_mae_val= []
    fold_rmse_val = []
    fold_nse_val = []
    fold_corr_val = []
    fold_r2_val = []
    fold_mae_train= []
    fold_rmse_train = []
    fold_nse_train = []
    fold_corr_train = []
    fold_r2_train = []
    models = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()  
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)       
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(1,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])               
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')   
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, 
                                       restore_best_weights=True)        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cts.NN_TEMP_WEIGHTS_DIR, "nn_epoch_{epoch}.keras"),
            save_weights_only=True,
            save_freq='epoch',
            verbose=0
        )   
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val_scaled),
            callbacks=[early_stopping, checkpoint_callback],
            verbose=0
        ) 
        y_pred_scaled = model.predict(X_train_scaled, verbose=0).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        mae = mean_absolute_error(y_train, y_pred)   
        rmse = math.sqrt(mean_squared_error(y_train, y_pred))
        nse = compute_nse(y_train,y_pred) 
        correlation_matrix = np.corrcoef(y_train.flatten(), y_pred.flatten())
        pearson_r = correlation_matrix[0, 1]
        r2 = pearson_r ** 2
        fold_mae_train.append(mae)  
        fold_rmse_train.append(rmse)
        fold_nse_train.append(nse)
        fold_corr_train.append(pearson_r)
        fold_r2_train.append(r2)    
        y_pred_scaled = model.predict(X_val_scaled, verbose=0).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        mae = mean_absolute_error(y_val, y_pred)   
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        nse = compute_nse(y_val,y_pred) 
        correlation_matrix = np.corrcoef(y_val.flatten(), y_pred.flatten())
        pearson_r = correlation_matrix[0, 1]
        r2 = pearson_r ** 2
        fold_mae_val.append(mae)  
        fold_rmse_val.append(rmse)
        fold_nse_val.append(nse)
        fold_corr_val.append(pearson_r)
        fold_r2_val.append(r2)
        models.append({
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'nse':nse
        })
    best_model_idx = np.argmax([m['nse'] for m in models])
    best_model = models[best_model_idx]
    results_train = {
        'mae': np.mean(fold_mae_train),
        'rmse': np.mean(fold_rmse_train),
        'nse': np.mean(fold_nse_train),
        'corr':np.mean(fold_corr_train),
        'r2': np.mean(fold_r2_train)
    } 
    results_val = {
        'mae': np.mean(fold_mae_val),
        'rmse': np.mean(fold_rmse_val),
        'nse': np.mean(fold_nse_val),
        'corr':np.mean(fold_corr_val),
        'r2': np.mean(fold_r2_val)
    }   
    return best_model,results_train ,results_val

def predict_discharge_fnn(model_path=cts.NN_WEIGHTS, scaler_pathx=cts.NN_SCALERSX,
                         scaler_pathy=cts.NN_SCALERSY, water_levels = None, 
                         return_as_array=False):
    if water_levels is None:
        return None
    model = keras.models.load_model(model_path)
    scaler_X = load_scaler_fnn(scaler_pathx)
    scaler_Y = load_scaler_fnn(scaler_pathy)
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
    
def build_polynomial_regression(kf, X, y, degree=3):
    fold_mae_val= []
    fold_rmse_val = []
    fold_nse_val = []
    fold_corr_val = []
    fold_r2_val = []
    fold_mae_train= []
    fold_rmse_train = []
    fold_nse_train = []
    fold_corr_train = []
    fold_r2_train = []
    models = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        poly_transformer = PolynomialFeatures(degree)
        model = make_pipeline(poly_transformer, LinearRegression())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        r2 = r2_score(y_train, y_pred)        
        mae = mean_absolute_error(y_train, y_pred)   
        rmse = math.sqrt(mean_squared_error(y_train, y_pred))
        nse = compute_nse(y_train,y_pred) 
        correlation_matrix = np.corrcoef(y_train.flatten(), y_pred.flatten())
        pearson_r = correlation_matrix[0, 1]
        r2 = pearson_r ** 2
        fold_mae_train.append(mae)  
        fold_rmse_train.append(rmse)
        fold_nse_train.append(nse)
        fold_corr_train.append(pearson_r)
        fold_r2_train.append(r2)
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)        
        mae = mean_absolute_error(y_val, y_pred)   
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        nse = compute_nse(y_val,y_pred) 
        correlation_matrix = np.corrcoef(y_val.flatten(), y_pred.flatten())
        pearson_r = correlation_matrix[0, 1]
        r2 = pearson_r ** 2
        fold_mae_val.append(mae)  
        fold_rmse_val.append(rmse)
        fold_nse_val.append(nse)
        fold_corr_val.append(pearson_r)
        fold_r2_val.append(r2)
        models.append({
            'model': model,
            'val_mse': rmse
        })
    best_model_idx = np.argmin([m['val_mse'] for m in models])
    best_model = models[best_model_idx]
    results_train = {
        'mae': np.mean(fold_mae_train),
        'rmse': np.mean(fold_rmse_train),
        'nse': np.mean(fold_nse_train),
        'corr':np.mean(fold_corr_train),
        'r2': np.mean(fold_r2_train)
    } 
    results_val = {
        'mae': np.mean(fold_mae_val),
        'rmse': np.mean(fold_rmse_val),
        'nse': np.mean(fold_nse_val),
        'corr':np.mean(fold_corr_val),
        'r2': np.mean(fold_r2_val)
    }   
    return best_model, results_train ,results_val

def get_poly_model_equation(model):
    lin_reg = model.named_steps["linearregression"]
    coeffs = lin_reg.coef_
    intercept = lin_reg.intercept_
    equation = f"y = {intercept:.2f} + {coeffs[1]:.2f}x + {coeffs[2]:.2f}x² + {coeffs[3]:.2f}x³"
    return equation

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

def build_random_forest(kf, X, y, n_estimators=100, max_depth=None):
    fold_mae_val= []
    fold_rmse_val = []
    fold_nse_val = []
    fold_corr_val = []
    fold_r2_val = []
    fold_mae_train= []
    fold_rmse_train = []
    fold_nse_train = []
    fold_corr_train = []
    fold_r2_train = []
    models = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
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
        model = grid_search.best_estimator_
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)        
        mae = mean_absolute_error(y_train, y_pred)   
        rmse = math.sqrt(mean_squared_error(y_train, y_pred))
        nse = compute_nse(y_train,y_pred) 
        correlation_matrix = np.corrcoef(y_train.flatten(), y_pred.flatten())
        pearson_r = correlation_matrix[0, 1]
        r2 = pearson_r ** 2
        fold_mae_train.append(mae)  
        fold_rmse_train.append(rmse)
        fold_nse_train.append(nse)
        fold_corr_train.append(pearson_r)
        fold_r2_train.append(r2)
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)        
        mae = mean_absolute_error(y_val, y_pred)   
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        nse = compute_nse(y_val,y_pred) 
        correlation_matrix = np.corrcoef(y_val.flatten(), y_pred.flatten())
        pearson_r = correlation_matrix[0, 1]
        r2 = pearson_r ** 2
        fold_mae_val.append(mae)  
        fold_rmse_val.append(rmse)
        fold_nse_val.append(nse)
        fold_corr_val.append(pearson_r)
        fold_r2_val.append(r2)
        models.append({
            'model': model,
            'val_rmse': rmse
        })
    best_model_idx = np.argmin([m['val_rmse'] for m in models])
    best_model = models[best_model_idx] 
    results_train = {
        'mae': np.mean(fold_mae_train),
        'rmse': np.mean(fold_rmse_train),
        'nse': np.mean(fold_nse_train),
        'corr':np.mean(fold_corr_train),
        'r2': np.mean(fold_r2_train)
    }    
    results_val = {
        'mae': np.mean(fold_mae_val),
        'rmse': np.mean(fold_rmse_val),
        'nse': np.mean(fold_nse_val),
        'corr':np.mean(fold_corr_val),
        'r2': np.mean(fold_r2_val)
    }   
    return best_model, results_train, results_val

def predict_discharge_forest(model_path=cts.RAND_WEIGHTS, water_levels=None, return_as_array=False):
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

def visualize_models_on_train_set(train_set):
  X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
  y_train = np.array(train_set["discharge"]).reshape(-1, 1)
  rf_train_preds = predict_discharge_forest(water_levels=X_train)
  poly_train_preds = predict_discharge_poly(water_levels=X_train)
  nn_train_preds = predict_discharge_fnn(water_levels=X_train)
  sorted_train_indices = np.argsort(X_train.flatten())  # Sort based on X_train (water level)
  X_train_sorted = X_train[sorted_train_indices]
  y_train_sorted = y_train[sorted_train_indices]
  rf_train_preds_sorted = [rf_train_preds[i] for i in sorted_train_indices]
  poly_train_preds_sorted = [poly_train_preds[i] for i in sorted_train_indices]
  nn_train_preds_sorted = [nn_train_preds[i] for i in sorted_train_indices]
  plt.figure(figsize=(7, 5))
  plt.scatter(X_train_sorted, y_train_sorted, color='gray', alpha=0.6, label="True Values")
  plt.plot(X_train_sorted, rf_train_preds_sorted, color='r', label='Random Forest', linewidth=2)
  plt.plot(X_train_sorted, poly_train_preds_sorted, color='g', label='Polynomial Regression', linewidth=2)
  plt.plot(X_train_sorted, nn_train_preds_sorted, color='b', label='Feed Foward Neural Network', linewidth=2)
  plt.xlabel("Water levels (m)")
  plt.ylabel(f"Discharge ($m^3$/s)")
  plt.legend(loc="best",fontsize=11)
  plt.grid(True)
  plt.savefig(cts.ALL_TRAIN_SET_EVALUATION_PLOT, dpi=300,bbox_inches='tight')
  plt.show()

def get_metrics(y_observed, y_pred):
    mae = mean_absolute_error(y_observed, y_pred)   
    rmse = math.sqrt(mean_squared_error(y_observed, y_pred))
    nse = compute_nse(y_observed,y_pred) 
    correlation_matrix = np.corrcoef(y_observed.flatten(), y_pred.flatten())
    pearson_r = correlation_matrix[0, 1]
    r2 = pearson_r ** 2
    return {
        'mae': mae,
        'rmse': rmse,
        'nse': nse,
        'corr': pearson_r,
        'r2': r2
    } 

def evaluate_models_on_test(test_set):
    X_test = np.array(test_set["waterlevel"]).reshape(-1, 1)
    y_test = np.array(test_set["discharge"]).reshape(-1, 1)
    rf_test_preds = predict_discharge_forest(water_levels=X_test)
    poly_test_preds = predict_discharge_poly(water_levels=X_test)
    nn_test_preds = predict_discharge_fnn(water_levels=X_test)
    all_values = np.concatenate([y_test, 
                               np.array(rf_test_preds).reshape(-1, 1),
                               np.array(poly_test_preds).reshape(-1, 1),
                               np.array(nn_test_preds).reshape(-1, 1)])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    rf_metrics = get_metrics(y_observed=y_test,y_pred=np.array(rf_test_preds).reshape(-1,1))
    poly_metrics = get_metrics(y_observed=y_test,y_pred=np.array(poly_test_preds).reshape(-1,1))
    fnn_metrics = get_metrics(y_observed=y_test,y_pred=np.array(nn_test_preds).reshape(-1,1))
    line_range = np.linspace(min_val, max_val, 100)   
    rf_model = LinearRegression()
    rf_model.fit(y_test, np.array(rf_test_preds).reshape(-1, 1))
    rf_line = rf_model.predict(line_range.reshape(-1, 1))
    plt.figure(figsize=(7, 6))
    rf_metric_r2 = rf_metrics['r2']
    poly_metric_r2 = poly_metrics['r2']
    fnn_metric_r2 = fnn_metrics['r2']
    plt.plot(line_range, rf_line, 'r', label=f'Random Forest: $R^2$ = {rf_metric_r2:.4f}', linewidth=2)
    poly_model = LinearRegression()
    poly_model.fit(y_test, np.array(poly_test_preds).reshape(-1, 1))
    poly_line = poly_model.predict(line_range.reshape(-1, 1))
    plt.plot(line_range, poly_line, 'g', label=f'Polynomial Regression: $R^2$ = {poly_metric_r2:.4f}', linewidth=2)
    nn_model = LinearRegression()
    nn_model.fit(y_test, np.array(nn_test_preds).reshape(-1, 1))
    nn_line = nn_model.predict(line_range.reshape(-1, 1))
    plt.plot(line_range, nn_line, 'b', label=f'Neural Network: $R^2$ = {fnn_metric_r2:.4f}', linewidth=2)
    plt.scatter(y_test, rf_test_preds, color='r', alpha=0.8, s=20)
    plt.scatter(y_test, poly_test_preds, color='g', alpha=0.8, s=20)
    plt.scatter(y_test, nn_test_preds, color='b', alpha=0.8, s=20)   
    plt.xlabel("Observed Discharge (m³/s)")
    plt.ylabel("Predicted Discharge (m³/s)")
    plt.legend(loc="best",fontsize=11)
    plt.grid(True)
    plt.savefig(cts.ALL_TEST_SET_EVALUATION_PLOT, dpi=300, bbox_inches='tight')
    plt.show()
    return fnn_metrics,poly_metrics,rf_metrics

def kfold_cross_validation_training(data_dict, n_splits=2):
    X = np.array(data_dict['waterlevel']).reshape(-1, 1)
    y = np.array(data_dict['discharge'])
    kf = KFold(n_splits=n_splits, shuffle=False)
    batch_size = int((len(X) / n_splits) * (n_splits - 1)) 
    print("Building FNN.......")
    ffnn_model, ffnn_train_results, ffnn_val_results = build_feedforward_nn(kf=kf,X=X,y=y,epochs=1000,batch_size=batch_size)
    ffnn_model['model'].save(cts.NN_WEIGHTS)
    save_scaler_fnn(ffnn_model['scaler_X'],cts.NN_SCALERSX)
    save_scaler_fnn(ffnn_model['scaler_y'],cts.NN_SCALERSY)
    os.makedirs(cts.NN_TEMP_WEIGHTS_DIR, exist_ok=True)
    df = pd.DataFrame({'Train Loss':ffnn_model['train_loss'],'Val Loss':ffnn_model['val_loss']})
    df.to_csv(cts.NN_LOGFILE,index=True)
    print("====FNN Evaluation Metrics======")
    print("Average training metrics  :===")
    dump_evaluation_metrics(ffnn_train_results)
    print("Average validation metrics:===")
    dump_evaluation_metrics(ffnn_val_results)
    print("Building Poly.......")
    poly_model, poly_train_results, poly_val_results = build_polynomial_regression(kf=kf,X=X,y=y)
    poly_equation = get_poly_model_equation(poly_model['model'])
    joblib.dump(poly_model['model'], cts.POLY_WEIGHTS) 
    print("====Poly Evaluation Metrics======")
    print("Average training metrics  :===")
    dump_evaluation_metrics(poly_train_results)
    print("Average validation metrics:===")
    dump_evaluation_metrics(poly_val_results)
    print(f"Poly Equation: {poly_equation}")  
    print("Building RandF......")
    rf_model, rf_train_results, rf_val_results = build_random_forest(kf=kf,X=X,y=y)
    joblib.dump(rf_model['model'], cts.RAND_WEIGHTS)
    print("====RandF Evaluation Metrics======")
    print("Average training metrics  :===")
    dump_evaluation_metrics(rf_train_results)
    print("Average validation metrics:===")
    dump_evaluation_metrics(rf_val_results)
    print(f"Saved FNN model to: {cts.NN_WEIGHTS}")
    print(f"Save FNN train and val loss history to: {cts.NN_LOGFILE}")
    print(f"Saved Poly model to: {cts.POLY_WEIGHTS}")
    print(f"Saved RandF model to: {cts.RAND_WEIGHTS}")
    return ffnn_val_results,poly_val_results,rf_val_results

def retrain_best_model(train_set,test_set,flag=1):
    if flag == 1:
        import random_forest as rf
        print('Training random forest on entire training set')
        rf.build_forest_model(train_set=train_set)
        rf.evaluate_forest_model(joblib.load(cts.RAND_WEIGHTS),train_set,test_set)       
    elif flag == 2:
        import poly_model as poly
        print("Training polynomial on entire training set")
        poly.build_poly_model(train_set=train_set)
        poly.evaluate_poly_model(joblib.load(cts.POLY_WEIGHTS),train_set,test_set)
    elif flag == 3:
        import neural_network as nn 
        print("Training FNN on entire training set")
        nn.build_nn_model(train_set=train_set)
        nn.evaluate_nn_model(keras.models.load_model(cts.NN_WEIGHTS),train_set,test_set)

def select_best_generalized_model(k=10):
    ffnn_val_results,poly_val_results,rf_val_results = \
    kfold_cross_validation_training(data_dict=train_set,n_splits=k)
    if (ffnn_val_results['rmse'] <= poly_val_results['rmse']) \
    and (ffnn_val_results['rmse'] <= rf_val_results['rmse']):
        print("Feed Forward Neural Network best generalizes")
        return 3
    elif (poly_val_results['rmse'] <= ffnn_val_results['rmse']) \
    and (poly_val_results['rmse'] <= rf_val_results['rmse']):
        print("Polynomial model best generalizes")
        return 2
    else:
        print("Random forest best generalizes")
        return 1

if __name__ == "__main__":
    [train_set,test_set] = dh.load_data()
    print(f"Train set length: {len(train_set['waterlevel'])}")
    print(f"Test set length: {len(test_set['discharge'])}")
    k = 10
    flag = select_best_generalized_model(k=k)
    print("Visualizing Models on train set")
    visualize_models_on_train_set(train_set=train_set)
    print(f"Best Model flag: {flag}")
    retrain_best_model(train_set=train_set,test_set=test_set,flag=flag)
    
    # print("Evaluating models on test set")
    # fnn_metrics,poly_metrics,rf_metrics =  evaluate_models_on_test(test_set=test_set)
    # print("====FNN Evaluation Metrics======")
    # print("Average test metrics  :===")
    # dump_evaluation_metrics(fnn_metrics)
    # print("====Poly Evaluation Metrics======")
    # print("Average test metrics  :===")
    # dump_evaluation_metrics(poly_metrics)
    # print("====RandF Evaluation Metrics======")
    # print("Average test metrics  :===")
    # dump_evaluation_metrics(rf_metrics)


    