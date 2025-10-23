from sklearn.model_selection import train_test_split

df = read

X = df.drop(['Weight'], axis=1)
y = df['Weight']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)
