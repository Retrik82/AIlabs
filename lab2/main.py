import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"C:\Users\Artsiom\PycharmProjects\AIlabs\data_all\processed_data.csv")
df.info()
df.head()

#We normalize the required fields for linear regression
scaler = MinMaxScaler()
df["age"] = scaler.fit_transform(df[["age"]])
df["bmi"] = scaler.fit_transform(df[["bmi"]])

df.head()

df['charges']

X = df.drop(['charges', 'id_number', 'children', 'sex_male', 'sex_female', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'], axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

linear_model = LinearRegression()
n = 6
poly_features = PolynomialFeatures(n)
X_train_poly = poly_features.fit_transform(X_train)
linear_model.fit(X_train_poly, y_train)
X_test_poly = poly_features.transform(X_test)
y_predict_test = linear_model.predict(X_test_poly)

MSE = mean_squared_error(y_test, y_predict_test)
RMSE = root_mean_squared_error(y_test, y_predict_test)
MAE = mean_absolute_error(y_test, y_predict_test)

print('Mean Square Error: ', MSE)
print('Root MSE: ', RMSE)
print('Mean Absolute Error: ', MAE)

lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)

ridge = Ridge(alpha=2.0)
ridge.fit(X_train, y_train)

elastic_model = ElasticNet(alpha=2.0, l1_ratio=0.5)
elastic_model.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)
ridge_pred = ridge.predict(X_test)
elastic_pred = elastic_model.predict(X_test)

print("Lasso MSE:", mean_squared_error(y_test, lasso_pred))
print("Ridge MSE:", mean_squared_error(y_test, ridge_pred))
print("Lasso + Ridge MSE:", mean_squared_error(y_test, elastic_pred))

df["charges"] = scaler.fit_transform(df[["charges"]])
X1 = df.drop(['smoker', 'id_number'], axis=1)
y1 = df["smoker"]

X_train1, X_temp1, y_train1, y_temp1 = train_test_split(
    X1, y1, test_size=0.4, random_state=42, stratify=y1)

X_val1, X_test1, y_val1, y_test1 = train_test_split(
    X_temp1, y_temp1, test_size=0.6, random_state=42, stratify=y_temp1)

logreg_model = LogisticRegression(max_iter = 2000)

logreg_model.fit(X_train1, y_train1)
y_pred_test = logreg_model.predict(X_test1)

accuracy = accuracy_score(y_test1, y_pred_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test1, y_pred_test)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

report = classification_report(y_test1, y_pred_test)
print(report)