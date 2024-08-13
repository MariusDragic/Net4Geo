import argparse
import joblib
import os
from dataprocessor import DataProcessor
from featureengineering import FeatureEngineering
from model import RandomForestModel

def parser():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser(description="Process to train or load model for predictions.")
    parser.add_argument('--process', type=str, default='load', choices=['load', 'train'],
                        help="Specify 'load' to load a pre-trained model, or 'train' to train a new model.")
    parser.add_argument('--train_data', type=str, default='train_100.geojson',
                        help="Path to the training data file inside the /dataset directory.")
    parser.add_argument('--test_data', type=str, default='test_100.geojson',
                        help="Path to the test data file inside the /dataset directory.")
    args = parser.parse_args()

    return args

def main(args):
    """
    Main function to load data, perform feature engineering, select best features,
    and either train a Random Forest model or load a pre-trained model to make predictions.
    """
    print('<<<< RUNNING MAIN PROGRAM >>>>')
    print('---- Loading geojson data and extracting geographical features ----')

    train_data_path = os.path.join('dataset', args.train_data)
    test_data_path = os.path.join('dataset', args.test_data)

    data_loader = DataProcessor(train_data_path, test_data_path)
    train_df, test_df, train_y = data_loader.load_and_prepare_data()

    print('---- Feature engineering module: creating new features and selecting best features ----')
    feature_engineer = FeatureEngineering(train_df, test_df)
    train_df_new, test_df_new = feature_engineer.process(train_y)

    model_path = os.path.join('model', 'random-forest.pkl')

    if args.process == 'train':
        print('---- Training the Random Forest model ----')
        model = RandomForestModel()
        model.train(train_df_new, train_y)

        os.makedirs('model', exist_ok=True)

        print('---- Saving the trained model in /model directory ----')
        with open(model_path, 'wb') as model_file:
            joblib.dump(model, model_file)

        model.predict_and_save(test_df_new, 'random_forest.csv')

    elif args.process == 'load':
        print('---- Loading the pre-trained model from /model directory ----')
        with open(model_path, 'rb') as model_file:
            model = joblib.load(model_file)

        model.predict_and_save(test_df_new, 'predictions.csv')


if __name__ == '__main__':
    args = parser()
    main(args)
