from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class RandomForestModel:
    """
    Class to handle the training and prediction using a Random Forest model.
    """

    def __init__(self, n_estimators=700, random_state=50):
        """
        Initialize RandomForestModel with specified parameters.

        :param n_estimators: Number of trees in the forest.
        :param random_state: Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, train_df, train_y):
        """
        Train the Random Forest model on the provided training data.

        :param train_df: Training DataFrame with features.
        :param train_y: Series with target variable.
        """
        self.model.fit(train_df, train_y)

    def predict_and_save(self, test_df, output_path):
        """
        Predict on the testing data and save the predictions to a CSV file.

        :param test_df: Testing DataFrame with features.
        :param output_path: File path to save the predictions.
        """
        predictions = self.model.predict(test_df)
        pred_df = pd.DataFrame(predictions, columns=['change_type'])
        pred_df.to_csv(output_path, index=True, index_label='Id')
