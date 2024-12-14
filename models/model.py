import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
from sklearn.utils import resample
import time

def feature_engineering(df):
    # Fill missing values if any
    df.fillna(df.median(), inplace=True)
    
    # Standardize numerical features
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    return df

def train_gbm_model(input_csv, solutions_csv, output_csv):
    print(f"Reading data from {input_csv}")
    data = pd.read_csv(input_csv)
    solutions = pd.read_csv(solutions_csv)

    print("Data read successfully. Preparing for training...")

    # Feature engineering (fill missing values, scale data, etc.)
    X = feature_engineering(data.drop(['vulnerability_name'], axis=1))
    y = data['vulnerability_name']

    # Handle class imbalance (if applicable)
    X, y = resample(X, y, replace=True, n_samples=len(y), random_state=42)
    
    # K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define the model
    model = lgb.LGBMClassifier()

    # Hyperparameter tuning
    param_grid = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 500],
        'max_depth': [-1, 10, 20]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    print("Best parameters found: ", grid_search.best_params_)

    # Train the model with best parameters
    best_model = grid_search.best_estimator_
    best_model.fit(X, y)
    print("Model training complete.")

    # Split the data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")

    # Save the model
    joblib.dump(best_model, 'models/gbm_model.joblib')
    print("Model saved to models/gbm_model.joblib")

    # Predict on the full data
    data['predicted_vulnerability'] = best_model.predict(X)

    # Merge with solutions
    final_data = pd.merge(data, solutions, left_on='predicted_vulnerability', right_on='vulnerability_name', how='left')

    # Save the final output
    final_data.to_csv(output_csv, index=False)
    print(f"Final output saved to {output_csv}")

if __name__ == "__main__":
    train_gbm_model('data/preprocessed_data.csv', 'data/solutions.csv', 'data/final_output.csv')
    
