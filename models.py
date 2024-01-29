from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def train_model(model_name, X_train, y_train):
  if model_name == "LinearModel":
    model = LinearRegression()
  elif model_name == "XGBoostModel":
    model = XGBRegressor()
  elif model_name == "rfModel":
    model = RandomForestRegressor()
  else:
    raise ValueError("Invalid model name")

  model.fit(X_train, y_train)
  return model