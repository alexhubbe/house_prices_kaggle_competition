from pathlib import Path

PALETTE = 'cividis'
RANDOM_STATE = 42
SCATTER_ALPHA = 0.2


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_FOLDER / "data"

# put the path for the project data files below
TRAIN_DATA = DATA_FOLDER / "train.csv" #original data from Kaggle
TEST_DATA = DATA_FOLDER / "test.csv" #original data from Kaggle
CLEAN_DATA = DATA_FOLDER / "clean_data.parquet"
TIDYMODEL_DATA = DATA_FOLDER / "ames_dataset.csv" # original data from R's tidymodel'MiscFeature'
PLUS_LON_LAT_DATA = DATA_FOLDER / "original_plus_lat_lon.csv" #Kaggle's data plus longitude and latitude from tidymodel
NAN_TREATMENT_MEDIAN_5_CLOSEST_SALEPRICE = DATA_FOLDER / 'nan_treatment_median_5_closest_saleprice.csv'
NAN_TO_FILL = DATA_FOLDER / 'nan_to_fill.csv'
CASE_SHILLER_DATA = DATA_FOLDER / 'CSUSHPINSA.csv' # data for the USA's Case Shiller Index
HOUSE_PRICE_INDEX = DATA_FOLDER / 'ATNHPIUS11180Q.csv' # data for the Ames' House Price Index

# put the path for the project model files below
MODELS_FOLDER = PROJECT_FOLDER / "models"

# put any other necessary paths below
REPORT_FOLDER = PROJECT_FOLDER / "reports"
IMAGES_FOLDER = REPORT_FOLDER / "images"

# put the path for the image files below:
MAP_PRICES_AND_NANS = IMAGES_FOLDER/'sale_prices_train_test_plot.png'
MAP_PRICES = IMAGES_FOLDER/'sale_prices.png'