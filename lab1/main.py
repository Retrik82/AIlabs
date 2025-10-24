import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

df = pd.read_csv("processed_test.csv")

#Task 1
print(df['FoodCourt'])

df.info()

X = df.drop(['FoodCourt', 'PassengerId', 'Name', 'Cabin'], axis=1)
y = df['FoodCourt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

#Task 2

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


#Task 3
y_predict_test = linear_model.predict(X_test)

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

elastic_model = ElasticNet(alpha=2.0, l1_ratio=0.1)
elastic_model.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)
ridge_pred = ridge.predict(X_test)
elastic_pred = elastic_model.predict(X_test)

print("Lasso MSE:", mean_squared_error(y_test, lasso_pred))
print("Ridge MSE:", mean_squared_error(y_test, ridge_pred))
print("Lasso + Ridge MSE:", mean_squared_error(y_test, elastic_pred))

#Task 4

y_class = (df['Spa'] > df['Spa'].median()).astype(int) #высокий, низкий расход
X1 = df.drop(['Spa', 'PassengerId', 'Name', 'Cabin'], axis=1)
y1 = y_class

X_train1, X_temp1, y_train1, y_temp1 = train_test_split(
    X1, y1, test_size=0.4, random_state=42, stratify=y1
)

X_val1, X_test1, y_val1, y_test1 = train_test_split(
    X_temp1, y_temp1, test_size=0.6, random_state=42, stratify=y_temp1
)

logreg_model = LogisticRegression()

logreg_model.fit(X_train1, y_train1)
y_pred_test = logreg_model.predict(X_test1)

#Task 5
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


#новый текст для реквеста
