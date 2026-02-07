import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = {
    'Apartment_Space': [600, 800, 1000, 1200, 1400, 1600, 2000, 2500],
    'Num_Rooms': [2, 2, 3, 3, 4, 4, 5, 5],
    'Price': [150000, 165000, 210000, 250000, 310000, 360000, 410000, 500000]
}

df = pd.DataFrame(data)

X = df[['Apartment_Space', 'Num_Rooms']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)

n_house = [[1750, 4]]
predicted_price = model.predict(n_house)

print(f"Predicted Price is: {int(predicted_price[0])}")
