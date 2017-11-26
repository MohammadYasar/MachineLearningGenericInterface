# Machine Learning Project - Credit Card Fraud Detection

# Structure:
+ folders - `src`, `test`, `data`, `utility`
+ `src` - all the classes for models
+ `test` - all the scripts for testing class
+ `data` - csv files
+ `utility` - raw files from individual

# Requirements:
### `XGBoost` package:
    pip install xgboost
or

    conda install -c conda-forge xgboost

### `imblearn` package:
    pip install -U imbalanced-learn
or

    conda install -c glemaitre imbalanced-learn

# Tutorial:
Here object oriented concept is used to implement any algorithm. For every
algorithm, one needs to take the base class `predictor` and then implement its
own version. For example, in `src` there are multiple implementations already.
For testing the algorithms test scripts are available in `test`.
