import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"C:\Users\Artsiom\PycharmProjects\AIlabs\data_all\diabetes.csv")

print(df.head())
print(df.info())
print(df.columns)

print('Missing values')
is_missing = df.isnull().sum()
print(is_missing)

X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)
rf_improved = RandomForestClassifier(
    n_estimators=300,                  # больше деревьев — лучше и стабильнее
    max_depth=None,                    # позволяем деревьям расти глубоко (переобучение нивелируется ансамблем)
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',               # по умолчанию для классификации, оптимально
    bootstrap=True,                    # обязательно для OOB
    oob_score=True,
    random_state=42,
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

y_proba_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = auc(fpr_rf, tpr_rf)

print("RF:")
print("OOB score for RF:", rf.oob_score_)
print(f"Accuracy in test: {accuracy_score(y_test, y_pred_rf):.4f}\n")

ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

ada_improved = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),  # глубина 3 вместо 1 даёт лучший результат на diabetes
    n_estimators=200,                               # больше итераций
    learning_rate=0.05,                             # меньший шаг — медленнее, но точнее
    random_state=42
)

ada.fit(X_train, y_train)

y_proba_ada = ada.predict_proba(X_test)[:, 1]
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_proba_ada)
auc_ada = auc(fpr_ada, tpr_ada)

y_pred_ada = ada.predict(X_test)
print(f"AdaBoost:")
print(f"Accuracy in test: {accuracy_score(y_test, y_pred_ada):.4f}\n")

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

gb_improved = GradientBoostingClassifier(
    n_estimators=300,                  # больше деревьев
    learning_rate=0.05,                # меньший learning_rate + больше деревьев = лучше
    max_depth=3,                       # деревья глубже (по умолчанию 3 — оптимально)
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,                     # стохастический бустинг — уменьшает переобучение
    max_features='sqrt',               # случайный выбор признаков, как в RF
    random_state=42
)

gb.fit(X_train, y_train)

y_proba_gb = gb.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
auc_gb = auc(fpr_gb, tpr_gb)

y_pred_gb = gb.predict(X_test)
print(f"Gradient Boosting:")
print(f"Accuracy in test: {accuracy_score(y_test, y_pred_gb):.4f}\n")

plt.figure(figsize=(9, 7))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})', linewidth=2.5)
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {auc_ada:.4f})', linewidth=2.5)
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {auc_gb:.4f})', linewidth=2.5)
plt.plot([0, 1], [0, 1], 'k--', label='Random classificator', alpha=0.7)

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC-curve', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Вывод финальных AUC
print(f"Final AUC:")
print(f"Random Forest:       {auc_rf:.4f}")
print(f"AdaBoost:            {auc_ada:.4f}")
print(f"Gradient Boosting:   {auc_gb:.4f}")