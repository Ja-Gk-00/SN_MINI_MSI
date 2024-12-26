import pandas as pd
import numpy as np
import os

import glob
import typing
from typing import Set
import csv

from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from textblob import TextBlob

DATA_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\data_raw"
DATA_TO_CONCATENATE = ["humidity.csv", "pressure.csv", "temperature.csv", "weather_description.csv", "wind_direction.csv", "wind_speed.csv"]
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\data_concatenated\\"
OUTPUT_DIR_FIN = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\data_concat_transformed\\"

def list_all_cities(path: str) -> Set[str]:

 with open(path) as csv_file:
    csv_reader = csv.DictReader(csv_file)
    dict_from_csv = dict(list(csv_reader)[0])
    list_of_column_names = list(dict_from_csv.keys())

    return list_of_column_names

def concatenate_city_data(cities, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(input_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in the directory: {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files. Processing...")
    
    for city in cities:
        print(f"\nProcessing city: {city}")
        merged_df = None 
        
        for file in csv_files:
            if str(file).split("\\")[-1] not in DATA_TO_CONCATENATE:
                continue
            try:
                df = pd.read_csv(file)
                
                if 'datetime' not in df.columns:
                    print(f"Warning: 'datetime' column not found in {file.name}. Skipping this file.")
                    continue
                
                if city not in df.columns:
                    print(f"Warning: City '{city}' not found in {file.name}. Skipping this city for this file.")
                    continue
                
                temp_df = df[['datetime', city]].copy()
                
                temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce')
                
                temp_df.dropna(subset=['datetime'], inplace=True)
                
                file_label = file.stem.lower()
                temp_df.rename(columns={city: file_label}, inplace=True)
                
                if merged_df is None:
                    merged_df = temp_df
                else:
                    merged_df = pd.merge(merged_df, temp_df, on='datetime', how='outer')
                    
            except Exception as e:
                print(f"Error processing file {file.name}: {e}")
        
        if merged_df is not None:
            merged_df.sort_values(by='datetime', inplace=True)
            
            merged_df.reset_index(drop=True, inplace=True)
            output_file = output_dir / f"{city}_concatenated.csv"
            
            try:
                merged_df.to_csv(output_file, index=False)
                print(f"Saved concatenated data for '{city}' to {output_file}")
            except Exception as e:
                print(f"Error saving data for city '{city}': {e}")
        else:
            print(f"No data found for city '{city}'. No file created.")


def preprocess_concatenated_data(
    df,
    normalize=True,
    categorize_weather=True,
    split_datetime=True,
    handle_missing=True,
    encode_categorical=True,
    feature_engineering=True
):

    processed_df = df.copy()

    if handle_missing:
        num_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
        num_imputer = SimpleImputer(strategy='median')
        processed_df[num_cols] = num_imputer.fit_transform(processed_df[num_cols])
        cat_cols = processed_df.select_dtypes(include=['object']).columns
        cat_imputer = SimpleImputer(strategy='most_frequent')
        processed_df[cat_cols] = cat_imputer.fit_transform(processed_df[cat_cols])

        print("Missing values handled.")

    if split_datetime:
        if 'datetime' in processed_df.columns:
            processed_df['date'] = pd.to_datetime(processed_df['datetime']).dt.date
            processed_df['time'] = pd.to_datetime(processed_df['datetime']).dt.time
            processed_df.drop('datetime', axis=1, inplace=True)
            print("'datetime' column split into 'date' and 'time'.")
        else:
            print("Warning: 'datetime' column not found. Skipping splitting step.")

    if categorize_weather:
        if 'weather_description' in processed_df.columns:
            processed_df['weather_sentiment'] = processed_df['weather_description'].apply(lambda x: TextBlob(x).sentiment.polarity)
            processed_df.drop('weather_description', axis=1, inplace=True)
            print("'weather_description' transformed into sentiment scores.")
        else:
            print("Warning: 'weather_description' column not found. Skipping sentiment transformation step.")

    if normalize:
        num_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
        scaler = MinMaxScaler()
        processed_df[num_cols] = scaler.fit_transform(processed_df[num_cols])
        print("Numerical columns normalized.")

    if encode_categorical:
        cat_cols = processed_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
        print("Categorical variables encoded.")

    if feature_engineering:
        if 'date' in processed_df.columns:
            processed_df['day_of_week'] = pd.to_datetime(processed_df['date']).dt.dayofweek
            print("Feature 'day_of_week' engineered.")

        if 'time' in processed_df.columns:
            processed_df['hour'] = processed_df['time'].astype(int)
            print("Feature 'hour' engineered.")

    return processed_df



def preprocess_and_save_concatenated_data(
    concatenated_dir,
    output_dir,
    normalize=True,
    categorize_weather=True,
    split_datetime=True,
    handle_missing=True,
    encode_categorical=True,
    feature_engineering=True
):

    concatenated_dir = Path(concatenated_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(concatenated_dir.glob('*.csv'))

    if not csv_files:
        print(f"No concatenated CSV files found in the directory: {concatenated_dir}")
        return

    print(f"Found {len(csv_files)} concatenated CSV files. Preprocessing...")

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"\nProcessing file: {file.name}")

            preprocessed_df = preprocess_concatenated_data(
                df,
                normalize=normalize,
                categorize_weather=categorize_weather,
                split_datetime=split_datetime,
                handle_missing=handle_missing,
                encode_categorical=encode_categorical,
                feature_engineering=feature_engineering
            )

            output_file = output_dir / f"{file.stem}_preprocessed.csv"

            preprocessed_df.to_csv(output_file, index=False)
            print(f"Saved preprocessed data to {output_file}")

        except Exception as e:
            print(f"Error processing file {file.name}: {e}")



### Main Script Run ###

unique_cities = set({})

for file in glob.glob(DATA_DIR + "\\*.csv"):
    if file.split("\\")[-1] not in DATA_TO_CONCATENATE:
       continue

    unique_cities.update(list_all_cities(file))

#concatenate_city_data(list(unique_cities), DATA_DIR, OUTPUT_DIR)

preprocess_and_save_concatenated_data(OUTPUT_DIR, OUTPUT_DIR_FIN)
