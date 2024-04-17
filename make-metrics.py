import os
import joblib
import pandas as pd
from utils import (
    get_player_name,
    prepare_data_ml,
)

data_dir = "data/"
models_dir = "models/"
for filename in os.listdir(data_dir):
    if filename.endswith(".parquet"):
        print("")
        print("")
        print(filename, "----------------------------------------------------------")
        print(get_player_name(int(filename.replace(".parquet", ""))))
        df = pd.read_parquet(os.path.join(data_dir, filename))
        _, _, X_test, y_test = prepare_data_ml(df)
        model_filename = filename.replace(".parquet", ".joblib")
        model = joblib.load(models_dir + model_filename)
        print(f"CV Accuracy:   {model.best_score_}")
        test_score = model.score(X_test, y_test)
        print(f"Test Accuracy: {test_score}")
        if test_score < 0.95:
            print("!!!!!!!!!!!!")
