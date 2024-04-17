import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from utils import (
    split_last_game,
    filter_rare_pitch_types,
    get_player_name,
    prepare_data_ml,
)


def train_model(X_train, y_train):

    # create pipeline for numeric features
    numeric_features = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z"]
    numeric_transformer = Pipeline(
        steps=[
            ("Median Imputer", SimpleImputer(strategy="median")),
            ("Standardization", StandardScaler()),
        ]
    )

    # create pipeline for categorical features
    categorical_features = ["stand"]
    categorical_transformer = Pipeline(
        steps=[
            ("Modal Imputer", SimpleImputer(strategy="most_frequent")),
            ("One-Hot Encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # create general preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("Numeric Transformer", numeric_transformer, numeric_features),
            ("Categorical Transformer", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # setup the full pipeline
    pipeline = Pipeline(
        steps=[
            ("Preprocessor", preprocessor),
            ("Classifier", LogisticRegression(multi_class="multinomial", max_iter=1000)),
        ]
    )

    # define the parameter grid for grid search
    param_grid = {
        "Classifier__C": [0.01, 0.1, 1.0, 10.0],
    }

    # setup grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")

    # perform grid search
    grid_search.fit(X_train, y_train)

    # return model
    return grid_search


data_dir = "data/"
models_dir = "models/"
for filename in os.listdir(data_dir):
    if filename.endswith(".parquet"):
        print("")
        print("")
        print(filename, "----------------------------------------------------------")
        print(get_player_name(int(filename.replace(".parquet", ""))))
        df = pd.read_parquet(os.path.join(data_dir, filename))
        X_train, y_train, X_test, y_test = prepare_data_ml(df)
        model = train_model(X_train, y_train)
        model_filename = filename.replace(".parquet", ".joblib")
        joblib.dump(model, models_dir + model_filename)
        print(f"CV Accuracy:   {model.best_score_}")
        print(f"Test Accuracy: {model.score(X_test, y_test)}")
