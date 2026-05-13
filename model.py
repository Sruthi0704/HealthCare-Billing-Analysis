import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_cost_model():
    df = pd.read_csv("data/final_healthcare_billing.csv")

    # Select features
    df = df[['Medical Condition', 'Admission Type', 'Length_of_Stay', 'Billing Amount']]

    # Encode categorical
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Billing Amount', axis=1)
    y = df['Billing Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model, X.columns