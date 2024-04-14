## README: Invoice Classification Program

### Overview
This program is designed to classify invoice entries as duplicates or non-duplicates based on various features extracted from invoice data. It uses the Python programming language and leverages the XGBoost machine learning algorithm, enhanced with SMOTE for handling imbalanced datasets. The program includes preprocessing of data, feature engineering with polynomial features, and optimization of model parameters through randomized search.

### Installation

#### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

#### Required Libraries
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib

#### Install Instructions
1. Ensure Python and pip are installed on your machine.
2. Install the required Python libraries using pip:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib
   ```

### Usage

#### Data Preparation
Ensure your invoice data is in a CSV file named `invoice_data.csv` with the appropriate columns (e.g., Amount, Date, Status, Vendor, Duplicate).

#### Running the Program
Execute the program by running the `model.py` script from your command line:
```bash
python model.py
```

#### Outputs
The program will output:
- Best hyperparameters for the XGBoost model.
- Performance metrics (Precision, Recall, F1-Score, ROC AUC).
- A confusion matrix.
- A precision-recall curve plot.

### Program Structure

- **Data Preprocessing**: Scales numeric features and converts categorical variables into dummy/indicator variables.
- **Feature Engineering**: Generates polynomial features to capture interactions between numeric features.
- **Model Training**: Uses an imbalanced-learn pipeline to apply SMOTE and train an XGBoost classifier.
- **Hyperparameter Tuning**: Utilizes RandomizedSearchCV to find the best model parameters.
- **Evaluation**: Assesses the model's performance using classification metrics and plots.

### Contributing
Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.

### License
TBD

### Contact
For support or to contact the developers, please send an email to tristynsanchez88@gmail.com.
