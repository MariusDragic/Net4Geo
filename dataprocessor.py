import geopandas as gpd
import pandas as pd

class DataProcessor:
    """
    Class to load and prepare geospatial data for modeling.
    """

    CHANGE_TYPE_MAP = {
        'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
    }

    def __init__(self, train_path, test_path):
        """
        Initialize DataLoader with paths to the training and testing data files.

        :param train_path: Path to the training data file.
        :param test_path: Path to the testing data file.
        """
        self.train_path = train_path
        self.test_path = test_path

    def load_and_prepare_data(self):
        """
        Load the training and testing data from the specified paths and prepare the target variable.

        :return: Tuple of train_df (GeoDataFrame), test_df (GeoDataFrame), and train_y (Series of target labels).
        """
        train_df = gpd.read_file(self.train_path, index_col=0)
        test_df = gpd.read_file(self.test_path, index_col=0)

        train_y = train_df['change_type'].apply(lambda x: self.CHANGE_TYPE_MAP[x])

        return train_df, test_df, train_y
