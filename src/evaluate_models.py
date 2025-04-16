import matplotlib.pyplot as plt
import numpy as np
import neural_network as neural
import poly_model as poly
import random_forest as forest
import dataset_handler as dh
import constants as cts
plt.rcParams.update({'font.size': 13})

def evaluate_models(train_set, test_set):
  # Extract water levels and discharge from the train and test sets
  X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
  y_train = np.array(train_set["discharge"]).reshape(-1, 1)
  X_test = np.array(test_set["waterlevel"]).reshape(-1, 1)
  y_test = np.array(test_set["discharge"]).reshape(-1, 1)
  # Predict with Random Forest
  rf_train_preds = forest.predict_discharge_forest(water_levels=X_train)
  rf_test_preds = forest.predict_discharge_forest(water_levels=X_test)
  # Predict with Polynomial Regression
  poly_train_preds = poly.predict_discharge_poly(water_levels=X_train)
  poly_test_preds = poly.predict_discharge_poly(water_levels=X_test)
  # Predict with Neural Network
  nn_train_preds = neural.predict_discharge_nn(water_levels=X_train)
  nn_test_preds = neural.predict_discharge_nn(water_levels=X_test)
  # Sort the lists of water levels for smooth plotting
  sorted_train_indices = np.argsort(X_train.flatten())  # Sort based on X_train (water level)
  sorted_test_indices = np.argsort(X_test.flatten())  # Sort based on X_test (water level)
  # Sort the prediction lists according to sorted indices
  X_train_sorted = X_train[sorted_train_indices]
  y_train_sorted = y_train[sorted_train_indices]
  rf_train_preds_sorted = [rf_train_preds[i] for i in sorted_train_indices]
  poly_train_preds_sorted = [poly_train_preds[i] for i in sorted_train_indices]
  nn_train_preds_sorted = [nn_train_preds[i] for i in sorted_train_indices]
  X_test_sorted = X_test[sorted_test_indices]
  y_test_sorted = y_test[sorted_test_indices]
  rf_test_preds_sorted = [rf_test_preds[i] for i in sorted_test_indices]
  poly_test_preds_sorted = [poly_test_preds[i] for i in sorted_test_indices]
  nn_test_preds_sorted = [nn_test_preds[i] for i in sorted_test_indices]
  # Plot results for training set
  plt.figure(figsize=(7, 5))
  plt.scatter(X_train_sorted, y_train_sorted, color='gray', alpha=0.6, label="True Values")
  plt.plot(X_train_sorted, rf_train_preds_sorted, color='r', label='Random Forest', linewidth=2)
  plt.plot(X_train_sorted, poly_train_preds_sorted, color='g', label='Polynomial Regression', linewidth=2)
  plt.plot(X_train_sorted, nn_train_preds_sorted, color='b', label='Neural Network', linewidth=2)
  plt.xlabel("Water levels (m)")
  plt.ylabel(f"Discharge ($m^3$/s)")
  plt.legend(loc="best",fontsize=11)
  plt.grid(True)
  plt.savefig(cts.ALL_TRAIN_SET_EVALUATION_PLOT, dpi=300,bbox_inches='tight')
  plt.show()
  # Plot results for test set
  plt.figure(figsize=(7, 5))
  plt.scatter(X_test_sorted, y_test_sorted, color='gray', alpha=0.6, label="True Values")
  plt.plot(X_test_sorted, rf_test_preds_sorted, color='r', label='Random Forest', linewidth=2)
  plt.plot(X_test_sorted, poly_test_preds_sorted, color='g', label='Polynomial Regression', linewidth=2)
  plt.plot(X_test_sorted, nn_test_preds_sorted, color='b', label='Neural Network', linewidth=2)
  plt.xlabel("Water levels (m)")
  plt.ylabel(f"Discharge ($m^3$/s)")
  plt.legend(loc="best",fontsize=11)
  plt.grid(True)
  plt.savefig(cts.ALL_TEST_SET_EVALUATION_PLOT,dpi=300,bbox_inches='tight')
  plt.show()

plt.rcParams.update({'font.size': 13})

def evaluate_models_2(train_set, test_set):
    # Extract water levels and discharge from the train and test sets
    X_train = np.array(train_set["waterlevel"]).reshape(-1, 1)
    y_train = np.array(train_set["discharge"]).reshape(-1, 1)
    X_test = np.array(test_set["waterlevel"]).reshape(-1, 1)
    y_test = np.array(test_set["discharge"]).reshape(-1, 1)
    print(len(X_test))
    # Predict with Random Forest
    rf_test_preds = forest.predict_discharge_forest(water_levels=X_test)
    
    # Predict with Polynomial Regression
    poly_test_preds = poly.predict_discharge_poly(water_levels=X_test)
    
    # Predict with Neural Network
    nn_test_preds = neural.predict_discharge_nn(water_levels=X_test)
    
    # Create figure
    plt.figure(figsize=(8, 5))
    
    # Calculate min and max values for all data
    all_values = np.concatenate([y_test, 
                               np.array(rf_test_preds).reshape(-1, 1),
                               np.array(poly_test_preds).reshape(-1, 1),
                               np.array(nn_test_preds).reshape(-1, 1)])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    # Calculate R-squared for each model
    rf_r2 = calculate_r2_score(y_test, rf_test_preds)
    poly_r2 = calculate_r2_score(y_test, poly_test_preds)
    nn_r2 = calculate_r2_score(y_test, nn_test_preds)
    # Create range for perfect prediction line
    line_range = np.linspace(min_val, max_val, 100)
    
    # Plot diagonal line (perfect prediction)
    #plt.plot(line_range, line_range, 'k--', label='Perfect Prediction')
    # Create linear regression lines for each model
    from sklearn.linear_model import LinearRegression
    
    # Random Forest regression line
    rf_model = LinearRegression()
    rf_model.fit(y_test, np.array(rf_test_preds).reshape(-1, 1))
    rf_line = rf_model.predict(line_range.reshape(-1, 1))
    plt.plot(line_range, rf_line, 'r', label=f'Random Forest: $R^2$ = {rf_r2:.4f}', linewidth=2)
    
    # Polynomial Regression line
    poly_model = LinearRegression()
    poly_model.fit(y_test, np.array(poly_test_preds).reshape(-1, 1))
    poly_line = poly_model.predict(line_range.reshape(-1, 1))
    plt.plot(line_range, poly_line, 'g', label=f'Polynomial Regression: $R^2$ = {poly_r2:.4f}', linewidth=2)
    
    # Neural Network line
    nn_model = LinearRegression()
    nn_model.fit(y_test, np.array(nn_test_preds).reshape(-1, 1))
    nn_line = nn_model.predict(line_range.reshape(-1, 1))
    plt.plot(line_range, nn_line, 'b', label=f'Neural Network: $R^2$ = {nn_r2:.4f}', linewidth=2)
    
    # Plot scatter points
    plt.scatter(y_test, rf_test_preds, color='r', alpha=0.8, s=20)
    plt.scatter(y_test, poly_test_preds, color='g', alpha=0.8, s=20)
    plt.scatter(y_test, nn_test_preds, color='b', alpha=0.8, s=20)
    
    plt.xlabel("True Discharge (m³/s)")
    plt.ylabel("Predicted Discharge (m³/s)")
    plt.legend(loc="best",fontsize=11)
    plt.grid(True)
    plt.savefig(cts.ALL_TEST_SET_EVALUATION_PLOT, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_r2_score(y_true, y_pred):
    """
    Calculate the coefficient of determination (R²) score.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    r2 : float
        R² score between -inf and 1.0 (1.0 being perfect prediction)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()  
    # Compute Pearson correlation coefficient
    correlation_matrix = np.corrcoef(y_true.flatten(), y_pred.flatten())
    pearson_r = correlation_matrix[0, 1]
    # Compute R² as the square of Pearson’s r
    r2 = pearson_r ** 2
    
    return r2
if __name__ == "__main__":
  [train_set,test_set] = dh.load_data()
  evaluate_models_2(train_set=train_set,test_set=test_set)
    