import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

data = pd.read_csv("C:\\Users\\PRAJNA\\Downloads\\housing.csv").dropna().drop_duplicates()

imputer = SimpleImputer(strategy='mean')
data['longitude'] = imputer.fit_transform(data[['longitude']])

df_housing = pd.read_csv("C:\\Users\\PRAJNA\\Downloads\\housing.csv")
df_additional = pd.read_csv("C:\\Users\\PRAJNA\\Downloads\\new.csv")
merged_df = pd.merge(df_housing, df_additional, on='longitude', how='inner')
print('Merged datasets:\n', merged_df)

categorical_cols = ['ocean_proximity']
data_transformed = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
numerical_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
scaler = StandardScaler()
data_transformed[numerical_cols] = scaler.fit_transform(data_transformed[numerical_cols])

data_transformed.to_csv('processed_data.csv', index=False)
print(data_transformed)
