import os
import pandas as pd
from pybaseball import playerid_lookup, playerid_reverse_lookup


# get number of missing values in a data frame
def count_missing_values(df):
    return df.isnull().sum()


# extract mlbam key
def get_mlbam_key(last, first):
    player = playerid_lookup(last, first)
    return player["key_mlbam"].values[0]


# get player name given id
def get_player_name(id):
    name_info = playerid_reverse_lookup([id])
    first = name_info["name_first"][0].title()
    last = name_info["name_last"][0].title()
    return first + " " + last


# process statcast data
def process_statcast_data(df):
    df = df[~df["game_type"].isin(["E", "S"])]
    df = df[~df["pitch_type"].isin(["PO"])]
    df = df.dropna(subset=["pitch_type"])
    target_column = "pitch_type"
    feature_columns = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z", "stand"]
    metadata_columns = "game_date"
    all_columns = [target_column] + feature_columns + [metadata_columns]
    df = df[all_columns]
    df["pitch_type"] = df["pitch_type"].astype("category")
    return df


# remove pitches that are thrown less than some threshold
def filter_rare_pitch_types(df, count=5):
    pitch_type_counts = df["pitch_type"].value_counts()
    df_filtered = df[df["pitch_type"].isin(pitch_type_counts[pitch_type_counts >= count].index)]
    return df_filtered


# split a dataframe of statcast data into two, one for last game, one for previous
def split_last_game(df):
    game_dates = df["game_date"].unique()
    last_game_date = game_dates.max()
    previous_games = df[df["game_date"] != last_game_date]
    current_game = df[df["game_date"] == last_game_date]
    return previous_games, current_game


# read in data from data directory and preview and check missing data
def check_data():
    data_dir = "data/"
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet"):
            print("")
            print("")
            print(filename, "----------------------------------------------------------")
            print("")
            df = pd.read_parquet(os.path.join(data_dir, filename))
            print(count_missing_values(df))
            print("")
            print(df)
            print("")


# function to obtain the pitch mix for a pitcher passed by id
def get_pitch_mix(pitcher_id):
    df = pd.read_parquet("data/" + str(pitcher_id) + ".parquet")
    mix = df["pitch_type"].value_counts() / len(df)
    return mix


# train-test split a data frame then also create X and y variants
def prepare_data_ml(df):

    # train-test split data based on most recent game
    df_train, df_test = split_last_game(df)
    # filter rare pitch types
    df_train = filter_rare_pitch_types(df_train)

    # create X and y for train
    X_train = df_train.drop("pitch_type", axis=1)
    y_train = df_train["pitch_type"]

    # create X and y for test
    X_test = df_test.drop("pitch_type", axis=1)
    y_test = df_test["pitch_type"]

    return X_train, y_train, X_test, y_test


# get list of pitchers with a learned model
def get_available_pitcher_ids():
    files = os.listdir("models")
    ids = [int(f.replace(".joblib", "")) for f in files if f.endswith(".joblib")]
    return ids
