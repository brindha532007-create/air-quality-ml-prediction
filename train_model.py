import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    print("--- DATA AUDIT ---")
    input_file = 'Dataset (Air Quality).csv'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        original_data = list(reader)
        
    print(f"1. Original number of records: {len(original_data)}")
    
    # Drop duplicates
    seen = set()
    unique_data = []
    for row in original_data:
        tup = tuple(row.values())
        if tup not in seen:
            seen.add(tup)
            unique_data.append(row)
            
    print(f"2. Number of duplicate records: {len(original_data) - len(unique_data)}")
    
    # Count missing values
    missing_counts = {col: 0 for col in columns}
    cleaned_data = []
    
    for row in unique_data:
        is_missing = False
        for col in columns:
            val = row[col].strip() if row[col] else ""
            if not val or val.lower() == 'na' or val == 'null':
                missing_counts[col] += 1
                is_missing = True
        
        if not is_missing:
            cleaned_data.append(row)

    print("3. Number of missing values in each column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"   {col}: {count}")

    removed_count = len(unique_data) - len(cleaned_data)
    print(f"\nRecords removed due to missing values: {removed_count}")
    print("Exact reason for removal: These rows contained missing values ('NA') in critical columns.")
    
    states = set(r['state'] for r in cleaned_data)
    cities = set(r['city'] for r in cleaned_data)
    stations = set(r['station'] for r in cleaned_data)
    pollutants = set(r['pollutant_id'] for r in cleaned_data)
    
    print(f"4. Number of valid records after cleaning: {len(cleaned_data)}")
    print(f"5. Number of states: {len(states)}")
    print(f"6. Number of cities: {len(cities)}")
    print(f"7. Number of monitoring stations: {len(stations)}")
    print(f"8. Number of different pollutants: {len(pollutants)}")

    # Save cleaned dataset
    with open('cleaned_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cleaned_data)
    print("\nSaved cleaned dataset to 'cleaned_dataset.csv'.")

    # --- MACHINE LEARNING MODEL ---
    print("\n--- MODEL TRAINING ---")
    
    X_numeric = []
    X_categorical = []
    y = []
    
    # Process features
    for row in cleaned_data:
        # Date parsing
        dt = datetime.strptime(row['last_update'], '%d-%m-%Y %H:%M')
        
        num_features = [
            float(row['latitude']),
            float(row['longitude']),
            float(row['pollutant_min']),
            float(row['pollutant_max']),
            dt.year,
            dt.month,
            dt.day,
            dt.hour
        ]
        
        cat_features = [
            row['state'],
            row['city'],
            row['station'],
            row['pollutant_id']
        ]
        
        X_numeric.append(num_features)
        X_categorical.append(cat_features)
        y.append(float(row['pollutant_avg']))
        
    X_numeric = np.array(X_numeric)
    y = np.array(y)
    
    # One-hot encoding
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_categorical)
    
    # Combine features
    X = np.hstack((X_numeric, X_cat_encoded))
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    print(f"\nTotal dataset records: {len(cleaned_data)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Number of features used (after encoding): {X_train.shape[1]}")

    joblib.dump(model, 'linear_regression_model.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    
    # Save some test predictions manually
    with open('test_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Actual Pollutant Average', 'Predicted Pollutant Average'])
        for actual, pred in zip(y_test, y_pred):
            writer.writerow([actual, round(pred, 2)])
            
    print("\nModel saved as 'linear_regression_model.pkl'.")
    print("Encoder saved as 'encoder.pkl'.")
    print("Test predictions saved as 'test_predictions.csv'.")

if __name__ == "__main__":
    main()
