import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/insurance.csv")

print(df.head())
print(df.info())

df['id_number'] = range(len(df))
df = df[['id_number', 'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']]
df.head()

print('Missing values')
is_missing = df.isnull().sum()
print(is_missing)

#If there were missing values, the mode, median, or mean could be filled in.
#for example:
#age_mean = df["Name"].mean()
#df["age"] = df["age"].fillna(age_mean)

#Normalization will be different for different model

df = pd.get_dummies(df, columns= ['region'])
df = pd.get_dummies(df, columns= ['sex'])
df['smoker'] = df['smoker'].map({'yes': True, 'no': False})
print(df.head())

df.to_csv(r"C:\Users\Artsiom\PycharmProjects\AIlabs\data_all\processed_data.csv", index=False)