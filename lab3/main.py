import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import (DecisionTreeClassifier,
                          DecisionTreeRegressor,
                          plot_tree)
from sklearn.metrics import (mean_squared_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             auc,
                             roc_curve)
from sklearn.metrics import r2_score

#Regression

df = pd.read_csv(r"C:\Users\Artsiom\PycharmProjects\AIlabs\data_all\processed_data.csv")
df.head(10)

df["bmi_smoker"] = df["bmi"] * df["smoker"]
df["age2"] = df["age"] ** 2
df["obese"] = (df["bmi"] > 30).astype(int)
df["has_children"] = (df["children"] > 0).astype(int)

X_cols_1 = ["age", "age2", 'bmi', "obese", "smoker", "has_children",  'children', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest', 'bmi_smoker']
y_col_1 = "charges"

print(df[y_col_1])

X_regression = df[X_cols_1]
y_regression = df[y_col_1]


X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_regression, y_regression, test_size = 0.3, random_state = 42)

regression_tree_model = DecisionTreeRegressor(
    max_depth=7,
    min_samples_split=20,
    min_samples_leaf=5,
    random_state=42)
regression_tree_model.fit(X_train_r, y_train_r)

y_predict_test_r = regression_tree_model.predict(X_test_r)

MSE = mean_squared_error(y_test_r, y_predict_test_r)
RMSE = root_mean_squared_error(y_test_r, y_predict_test_r)
MAE = mean_absolute_error(y_test_r, y_predict_test_r)

print('Mean Square Error: ', MSE)
print('Root MSE: ', RMSE)
print('Mean Absolute Error: ', MAE)

R2 = r2_score(y_test_r, y_predict_test_r)
print('R-squared: ', R2)

plt.figure(figsize = (10, 6))
plot_tree(regression_tree_model, filled = True, feature_names = X_regression.columns)
plt.title("Дерево решений (Регрессия bmi)")
plt.show()

X_cols_2 = ["age", "bmi", "children", "charges",
    "region_northeast", "region_northwest",
    "region_southeast", "region_southwest",
    "sex_female", "sex_male"
]
y_col_2 = "smoker"

X_classification = df[X_cols_2]
y_classification = df[y_col_2]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_classification, y_classification, test_size = 0.3, random_state = 42)

classification_tree_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=10,
    random_state=42
)
classification_tree_model.fit(X_train_c, y_train_c)

y_probability = classification_tree_model.predict_proba(X_test_c)[:, 1]
print(classification_tree_model.classes_)
fpr, tpr, thresholds = roc_curve(y_test_c, y_probability)
auc_metric = auc(fpr, tpr)

print("ROC-AUC metric: ", auc_metric)

plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
plt.show()
