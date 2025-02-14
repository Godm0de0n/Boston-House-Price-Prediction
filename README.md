# Boston House Price Prediction using XGBoost

## Overview
This project implements a **Boston House Price Prediction** model using **XGBoost**. The goal is to predict house prices based on various features such as crime rate, number of rooms, and property tax.

## Dataset
The dataset used for this project is the **Boston Housing dataset**, which contains **506 samples** with **13 feature columns**. The dataset includes features such as:
- **CRIM** (Per capita crime rate by town)
- **ZN** (Proportion of residential land zoned for large lots)
- **INDUS** (Proportion of non-retail business acres per town)
- **CHAS** (Charles River dummy variable)
- **NOX** (Nitrogen oxide concentration)
- **RM** (Average number of rooms per dwelling)
- **AGE** (Proportion of owner-occupied units built before 1940)
- **DIS** (Weighted distances to employment centers)
- **RAD** (Index of accessibility to highways)
- **TAX** (Property tax rate per $10,000)
- **PTRATIO** (Pupil-teacher ratio by town)
- **B** (Proportion of Black residents)
- **LSTAT** (Lower status of the population)

You can download the dataset from the **UCI Machine Learning Repository** or from **Scikit-learn's built-in datasets**.
`house_price_dataset = sklearn.datasets.load_boston()`
will not work

# Why??
The Boston housing prices dataset has an ethical problem: as
investigated in [1], the authors of this dataset engineered a
non-invertible variable "B" assuming that racial self-segregation had a
positive impact on house prices [2]. Furthermore the goal of the
research that led to the creation of this dataset was to study the
impact of air quality but it did not give adequate demonstration of the
validity of this assumption.

The scikit-learn maintainers therefore strongly discourage the use of
this dataset unless the purpose of the code is to study and educate
about ethical issues in data science and machine learning.

In this special case, you can fetch the dataset from the original
source::

    ```import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]```

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Model Training & Evaluation
1. Load the dataset and preprocess the data.
2. Split the data into training and testing sets.
3. Train the **XGBoost** model.
4. Evaluate the model using MAE, R² score, and visualization of actual vs. predicted prices.

## Results
- Model performance metrics such as **MAE, and R² score** will be displayed after execution.
- Visualization of predicted vs. actual house prices.

## Contribution
Feel free to contribute by opening an issue or submitting a pull request.

## License
This project is licensed under the MIT License.
