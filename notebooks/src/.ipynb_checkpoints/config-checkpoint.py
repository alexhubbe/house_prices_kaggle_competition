from pathlib import Path

PALETTE = 'cividis'
RANDOM_STATE = 42
SCATTER_ALPHA = 0.2

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_FOLDER / "data"

# Define paths for project data files
TRAIN_DATA = DATA_FOLDER / "train.csv"  # Original train dataset from Kaggle
TEST_DATA = DATA_FOLDER / "test.csv"  # Original test dataset from Kaggle
CLEAN_DATA = DATA_FOLDER / "clean_data.csv"  # Dataset prepared for machine learning analysis
TIDYMODEL_DATA = DATA_FOLDER / "ames_dataset.csv"  # Original dataset from R's tidymodels
PLUS_LON_LAT_DATA = DATA_FOLDER / "original_plus_lat_lon.csv"  # Kaggle dataset with added longitude and latitude from tidymodels
CASE_SHILLER_DATA = DATA_FOLDER / 'CSUSHPINSA.csv'  # Data for the USA's Case-Shiller Index
HOUSE_PRICE_INDEX = DATA_FOLDER / 'ATNHPIUS11180Q.csv'  # Data for Ames' House Price Index

MODELS_FOLDER = PROJECT_FOLDER / "models"

REPORTS_FOLDER = PROJECT_FOLDER /"reports"
KAGGLE_SUBMISSION = REPORTS_FOLDER/"best_kaggle_prediction.csv"


IMAGES_FOLDER = REPORTS_FOLDER / "images"
# Define paths for image files
MAP_PRICES_AND_NANS = IMAGES_FOLDER / 'sale_prices_train_test_plot.png'
MAP_PRICES = IMAGES_FOLDER / 'sale_prices.png'


