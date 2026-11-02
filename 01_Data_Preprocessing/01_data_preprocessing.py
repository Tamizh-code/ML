import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

print("DATA PREPROCESSING")

# Load dataset
d = load_iris(as_frame=True)
df = d.frame

# Add categorical column
df['Species'] = d.target_names[d.target]

# Insert missing value for demonstration
df.loc[0, 'sepal length (cm)'] = None

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encoding
df['Species'] = LabelEncoder().fit_transform(df['Species'])

# Normalization
df.iloc[:, :4] = MinMaxScaler().fit_transform(df.iloc[:, :4])

# Display output
print("\nPreprocessed Dataset:")
print(df.head().to_string())

print("\nData preprocessing completed!")
# # Experiment 1: Missing values, categorical encoding, and numerical normalization
# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# data = load_breast_cancer(as_frame=True)
# df = data.frame.copy()
# df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
# df["age_group"] = np.where(df["mean radius"] > df["mean radius"].median(), "high", "low")
# df = df.drop(columns=["target"])

# # Add a few missing values to demonstrate missing-value handling.
# df.loc[df.index[:8], "mean radius"] = np.nan
# df.loc[df.index[8:14], "age_group"] = np.nan

# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# num_imputer = SimpleImputer(strategy="median")
# cat_imputer = SimpleImputer(strategy="most_frequent")
# df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
# df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# encoded = encoder.fit_transform(df[categorical_cols])
# encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
# scaler = StandardScaler()
# scaled = scaler.fit_transform(df[numeric_cols])
# scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

# processed = pd.concat([scaled_df, encoded_df], axis=1)
# print("Original shape:", df.shape)
# print("Processed shape:", processed.shape)
# print(processed.head())
