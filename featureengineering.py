import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


class FeatureEngineering:
    """
    Class to handle feature engineering and feature selection processes.
    """

    def __init__(self, train_df, test_df):
        """
        Initialize with training and testing dataframes.

        :param train_df: Training GeoDataFrame.
        :param test_df: Testing GeoDataFrame.
        """
        self.train_df = train_df
        self.test_df = test_df
        self.quantitative_features = None
        self.categorical_features = None

    def calculate_geometric_features(self):
        """
        Calculate area and perimeter for the training and testing datasets.
        Reproject the GeoDataFrames to a projected CRS for accurate calculations.
        """

        projected_crs = "EPSG:32633"

        self.train_df = self.train_df.to_crs(projected_crs)
        self.test_df = self.test_df.to_crs(projected_crs)

        self.train_df['area'] = self.train_df['geometry'].area
        self.train_df['perimeter'] = self.train_df['geometry'].length

        self.test_df['area'] = self.test_df['geometry'].area
        self.test_df['perimeter'] = self.test_df['geometry'].length

    def calculate_date_differences(self):
        """
        Convert date columns to datetime and calculate the differences between consecutive dates.
        """
        date_columns = ['date0', 'date1', 'date2', 'date3', 'date4']
        for col in date_columns:
            self.train_df[col] = pd.to_datetime(self.train_df[col], format='%d-%m-%Y')
            self.test_df[col] = pd.to_datetime(self.test_df[col], format='%d-%m-%Y')

        for i in range(1, len(date_columns)):
            self.train_df[f'diff_date_{i}'] = (
                        self.train_df[date_columns[i]] - self.train_df[date_columns[i - 1]]).abs().dt.days
            self.test_df[f'diff_date_{i}'] = (
                        self.test_df[date_columns[i]] - self.test_df[date_columns[i - 1]]).abs().dt.days

    def define_features(self):
        """
        Define and separate the categorical and quantitative features.
        """
        self.categorical_features = ['urban_type', 'geography_type', 'change_status_date0',
                                     'change_status_date1', 'change_status_date2',
                                     'change_status_date3', 'change_status_date4']

        self.quantitative_features = ['area', 'perimeter', 'diff_date_1', 'diff_date_2', 'diff_date_3', 'diff_date_4',
                                      'img_red_mean_date1', 'img_green_mean_date1', 'img_blue_mean_date1',
                                      'img_red_std_date1', 'img_green_std_date1', 'img_blue_std_date1',
                                      'img_red_mean_date2', 'img_green_mean_date2', 'img_blue_mean_date2',
                                      'img_red_std_date2', 'img_green_std_date2', 'img_blue_std_date2',
                                      'img_red_mean_date3', 'img_green_mean_date3', 'img_blue_mean_date3',
                                      'img_red_std_date3', 'img_green_std_date3', 'img_blue_std_date3',
                                      'img_red_mean_date4', 'img_green_mean_date4', 'img_blue_mean_date4',
                                      'img_red_std_date4', 'img_green_std_date4', 'img_blue_std_date4',
                                      'img_red_mean_date5', 'img_green_mean_date5', 'img_blue_mean_date5',
                                      'img_red_std_date5', 'img_green_std_date5', 'img_blue_std_date5']

    def fill_missing_values(self):
        """
        Fill missing values in quantitative features with the mean of the respective column.
        """
        self.train_df[self.quantitative_features] = self.train_df[self.quantitative_features].fillna(
            self.train_df[self.quantitative_features].mean())
        self.test_df[self.quantitative_features] = self.test_df[self.quantitative_features].fillna(
            self.test_df[self.quantitative_features].mean())

    def replace_infinite_values(self):
        """
        Replace infinite values with NaN.
        :return:
        """
        self.train_df = self.train_df.replace([np.inf, -np.inf], np.nan)
        self.test_df = self.test_df.replace([np.inf, -np.inf], np.nan)

    def encode_categorical_features(self):
        """
        One-hot encode the categorical features and align the training and testing datasets.
        """
        encoded_train_df = pd.get_dummies(self.train_df[self.categorical_features])
        encoded_test_df = pd.get_dummies(self.test_df[self.categorical_features])

        self.train_df = pd.concat([self.train_df[self.quantitative_features], encoded_train_df], axis=1)
        self.test_df = pd.concat([self.test_df[self.quantitative_features], encoded_test_df], axis=1)

        # Ensure both train and test datasets have the same columns
        self.train_df, self.test_df = self.train_df.align(self.test_df, join='inner', axis=1)

    def select_best_features(self, train_y):
        """
        Select the best features using the chi-square test.

        :param train_y: Series with target variable.
        :return: Tuple of DataFrames with selected features for training and testing sets.
        """
        selector = SelectKBest(chi2, k="all")
        selector.fit(self.train_df, train_y)

        summary_stats = pd.DataFrame({
            "input_variable": self.train_df.columns,
            "p_value": selector.pvalues_,
            "chi2_score": selector.scores_
        })

        # Define thresholds
        p_value_threshold = 0.05
        score_threshold = 5

        # Select features based on thresholds
        selected_features = summary_stats.loc[(summary_stats["chi2_score"] >= score_threshold) &
                                              (summary_stats["p_value"] <= p_value_threshold),
        "input_variable"].tolist()

        # Return the DataFrames with selected features only
        return self.train_df[selected_features], self.test_df[selected_features]

    def process(self, train_y):
        """
        Full feature engineering pipeline: calculating features, handling missing values,
        encoding categorical features, and selecting the best features.

        :param train_y: Series with target variable.
        :return: Tuple of processed DataFrames (train and test).
        """

        print('#### Calculating geometric features ####')
        self.calculate_geometric_features()
        print('#### Calculating date differences ####')
        self.calculate_date_differences()
        print('#### Replacing infinite values ####')
        self.replace_infinite_values()
        print('#### Defining features ####')
        self.define_features()
        print('#### Managing missing values ####')
        self.fill_missing_values()
        print('#### Encoding categorical features ####')
        self.encode_categorical_features()
        print('#### Selecting best features ####')
        return self.select_best_features(train_y)
