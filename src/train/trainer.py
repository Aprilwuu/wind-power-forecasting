from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

def train_model(df, target_col, config):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"]
    )
    
    model = LGBMRegressor(
        learning_rate = config["train"]["learning_rate"],
        n_estimators = config["train"]["n_estimators"]
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model training completed. R² score:{score:.4f}")
    return model
