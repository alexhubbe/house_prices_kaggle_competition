import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from geopy.distance import geodesic


# Function to calculate geodesic distance
def calculate_distance(row, point):
    return geodesic((row['Latitude'], row['Longitude']), point).kilometers

# Function to calculate the median sale price of the n closest houses excluding NaNs
def calculate_median_price(row_index, df, n=5):
    row = df.iloc[row_index]
    distances = []
    for index, other_row in df.iterrows():
        if row_index != index:  # Skip itself
            distance = calculate_distance(row, (other_row['Latitude'], other_row['Longitude']))
            distances.append((distance, other_row['SalePrice']))
    distances.sort(key=lambda x: x[0])
    closest_prices = [price for _, price in distances[:n] if not pd.isna(price)]
    return np.median(closest_prices) if closest_prices else np.nan

# Wrapper for parallel processing
def process_median_price(row_index, df, n):
    return calculate_median_price(row_index, df, n)

def calculate_all_medians(df, n=5):
    with ProcessPoolExecutor() as executor:
        # Use partial to fix `df` and `n` for each row index
        process_func = partial(process_median_price, df=df, n=n)
        results = list(executor.map(process_func, range(len(df))))
    df['Median_n_Closest_SalePrice'] = results
    return df
