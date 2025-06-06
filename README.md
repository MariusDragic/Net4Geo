# Geographical Area Classifier

## Overview

**Geographical Area Classifier** is a machine learning project aimed at classifying geographical areas based on various types of changes, such as demolition, road construction, residential development, and more. The project utilizes geospatial data and employs a Random Forest classifier to predict these changes based on features extracted from the data. please reade geographical-area-classifier.pdf document fore more accurate details.

Train and test dataset are too heavy to be uploaded on github. You can download them from the following links: https://www.kaggle.com/competitions/2el1730-machine-learning-project-january-2024/overview. 

The project was developed as part of a Kaggle competition and achieved a score of `0.86120` on the validation dataset of Kaggle's CentraleSuélec competetion.

A detailed report including theoretical background and extended results is available [here](https://github.com/MariusDragic/Net4Geo/blob/main/Net4Geo.pdf) for interested readers.

## Project Structure

```plaintext
geographical-area-classifier/
│
├── dataset/                       # Directory containing the GeoJSON datasets (train and test files)
├── model/                         # Directory to save/load the trained model
├── main.py                        # Main script for executing the project
├── dataprocessor.py               # Module for loading and processing data
├── featureengineering.py          # Module for feature engineering
├── model.py                       # Module for training and prediction using the Random Forest model
├── requirements.txt               # File containing the project dependencies
└── README.md                      # This README file
```

## Installation

### Prerequisites

Ensure that you have the following installed:

- Python 3.8 or higher
- Git

### Installing Dependencies

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/MariusDragic/geographical-area-classifier.git
    cd geographical-area-classifier
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use \`venv\Scripts\activate\`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model on the provided training data and save the results, use the following command:

```bash
python main.py --process train --train_data train_100.geojson --test_data test_100.geojson
```

This command will:
- Load the training and test data from the `dataset` directory.
- Perform feature engineering and select the best features.
- Train a Random Forest model on the training data.
- Save the trained model in the `model` directory.
- Generate predictions on the test data and save them to `predictions.csv`.

Note that `train_100.geojson` and `test_100.geojson` are reduced dataset to run faster the progam. You can use the full dataset `train.geojson` and `test.geojson` to train the model on the full dataset.
### Loading a Pre-trained Model for Prediction

If you have already trained a model and want to use it for prediction on new test data, use the following command:

```bash
python main.py --process load --train_data train_100.geojson --test_data test_100.geojson
```

This command will:
- Load the training and test data from the `dataset` directory.
- Perform feature engineering and select the best features.
- Load the pre-trained Random Forest model from the `model` directory.
- Generate predictions on the test data and save them to `predictions.csv`.

### Command-Line Arguments

- `--process`: Specifies the process to execute. Use `train` to train a new model or `load` to load a pre-trained model for predictions. Default is \`load\`.
- `--train_data`: Path to the training data file within the `dataset` directory. Default is `train_100.geojson`.
- `--test_data`: Path to the test data file within the `dataset` directory. Default is `test_100.geojson`.

## Results and Performance

The Random Forest model was evaluated on a validation set and achieved a score of `0.86120` on CentraleSupélec Kaggle's project. This score reflects the model's accuracy in classifying the various types of geographical changes.

## Contributing

Contributions are welcome! If you wish to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch `git checkout -b feature/new-feature`.
3. Make your changes and commit them `git commit -am 'Add new feature'`.
4. Push to the branch `git push origin feature/new-feature`.
5. Open a Pull Request.

## Author

**Marius Dragic**

For any questions, please contact me here [marius.dragic@gmail.com](mailto:marius.dragic@gmail.com) or [marius.dragic@student-cs.fr](mailto:marius.dragic@student-cs.fr).


