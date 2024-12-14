import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def preprocess_data(input_csv, output_csv):
    print(f"Reading data from {input_csv}")
    data = pd.read_csv(input_csv)
    print("Data read successfully. Columns:", data.columns)

    print("Applying transformations...")

    numeric_features = ['subnet_size', 'port', 'response_time']
    categorical_features = ['protocol', 'firewall_enabled', 'security_features']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    transformed_data = preprocessor.fit_transform(data.drop(['ipv6_address', 'vulnerability_name'], axis=1))

    transformed_df = pd.DataFrame(transformed_data, columns=np.hstack([
        numeric_features,
        preprocessor.transformers_[1][1]['encoder'].get_feature_names_out(categorical_features)
    ]))
    
    transformed_df['vulnerability_name'] = data['vulnerability_name']

    transformed_df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_data('data/ipv6_data.csv', 'data/preprocessed_data.csv')
