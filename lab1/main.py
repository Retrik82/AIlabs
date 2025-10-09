import pandas as pd

from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("test.csv")
print('--------------------------------------------------------------------------')

print(df.head(10))
print('--------------------------------------------------------------------------')
df.info()
print('---------------------------------------------------')
types = df.dtypes
print(types)
print('---------------------------------------------------')
print('Пропущенные значения')
is_missing = df.isnull().sum()
print(is_missing)
print('---------------------------------------------------')
home_mod = df["HomePlanet"].mode()[0]
cryo_sleep_mod = df["CryoSleep"].mode()[0]
cabin_mod = df["Cabin"].mode()[0]
destination_mod = df["Destination"].mode()[0]
vip_mod = df["VIP"].mode()[0]
name_mod = df["Name"].mode()[0]

age_med = df["Age"].median()
room_service_med = df["RoomService"].median()
food_court_med = df["FoodCourt"].median()
shopping_mall_med = df["ShoppingMall"].median()
spa_med = df["Spa"].median()
vr_deck_med = df["VRDeck"].median()


df["HomePlanet"] = df["HomePlanet"].fillna(home_mod)
df["CryoSleep"] = df["CryoSleep"].fillna(cryo_sleep_mod)
df["Cabin"] = df["Cabin"].fillna(cabin_mod)
df["Destination"] = df["Destination"].fillna(destination_mod)
df["VIP"] = df["VIP"].fillna(vip_mod)
df["Name"] = df["Name"].fillna(name_mod)

df["Age"] = df["Age"].fillna(age_med)
df["RoomService"] = df["RoomService"].fillna(room_service_med)
df["FoodCourt"] = df["FoodCourt"].fillna(food_court_med)
df["ShoppingMall"] = df["ShoppingMall"].fillna(shopping_mall_med)
df["Spa"] = df["Spa"].fillna(spa_med)
df["VRDeck"] = df["VRDeck"].fillna(vr_deck_med)

print('---------------------------------------------------')
print('Пропущенные значения')
missing_values = df.isnull().sum()
print(missing_values)

print('---------------------------------------------------')
scaler = MinMaxScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
df["RoomService"] = scaler.fit_transform(df[["RoomService"]])
df["FoodCourt"] = scaler.fit_transform(df[["FoodCourt"]])
df["ShoppingMall"] = scaler.fit_transform(df[["ShoppingMall"]])
df["Spa"] = scaler.fit_transform(df[["Spa"]])
df["VRDeck"] = scaler.fit_transform(df[["VRDeck"]])

print('---------------------------------------------------')
columns_to_transform = ["Destination", "HomePlanet"]
df = pd.get_dummies(df, columns=columns_to_transform)

print('---------------------------------------------------')
print(df.head(10))

df.to_csv("processed_test.csv", index=False)

