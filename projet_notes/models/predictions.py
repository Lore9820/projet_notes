import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict(df:pd.DataFrame):
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    predict(pd.DataFrame({"feature1":[1], "feature2":[2], "feature3":[3]}))

    from models.read_files import *
    from controllers.preprocessing import *

    logs = get_logs()
    logs = split_columns(logs)
    df = creer_df(logs)
    df = df_transformer(df)
    df = df_aligned(df, expected_columns=model.get_params()["features"])

    print(predict(df))