import os
import pandas as pd
from pybaseball import playerid_lookup


# get number of missing values in a data frame
def count_missing_values(df):
    return df.isnull().sum()


# extract mlbam key
def get_mlbam_key(last, first):
    player = playerid_lookup(last, first)
    return player["key_mlbam"].values[0]


# process statcast data
def process_statcast_data(df):
    df = df[~df["game_type"].isin(["E", "S"])]
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
