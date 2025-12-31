# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score

# def train_model(df):
#     features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Return', 'sentiment']
#     X = df[features]
#     y = df['Target']

#     if len(df) < 10:
#         raise ValueError("Not enough data to train the model.")

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#     model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Model Accuracy: {acc*100:.2f}%")

#     return model, X_test, y_test
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_model(df):
    # -------- Features (NO sentiment here) --------
    features = [
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'MA5',
        'MA10',
        'Return'
    ]

    X = df[features]
    y = df['Target']

    if len(df) < 30:
        raise ValueError("Not enough data to train the model.")

    # -------- Train / Test split (time-series safe) --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # -------- XGBoost Model --------
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # -------- Accuracy --------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc*100:.2f}%")

    return model, X_test, y_test
