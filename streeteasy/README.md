# Codecademy + StreetEasy

Radu's contributions: 
Implemented multiple linear regression for rent prediction.
Designed data visualization for better interpretability.
Evaluated model performance using R-squared scores.

Data Processing: Reads the dataset into a pandas DataFrame and preprocesses features.
Model Training: Uses scikit-learn's LinearRegression model to predict rental prices.
Data Visualization:
Scatter plots for actual vs. predicted rents.
Box plots for categorical features vs. rent.
Scatter plots for continuous features vs. rent.
Performance Evaluation: Computes the R-squared score for both training and test datasets.

Technologies Used:

Python

Pandas

NumPy

Scikit-learn

Seaborn

Matplotlib


https://www.codecademy.com/content-items/d19f2f770877c419fdbfa64ddcc16edc

[StreetEasy](www.streeteasy.com) is New York City's leading real estate marketplace — from studios to high-rises, Brooklyn Heights to Harlem.

In the [Multiple Linear Regression](https://www.codecademy.com/courses/multiple-linear-regression/lessons/multiple-linear-regression-streeteasy/exercises/introduction) (MLR) lesson, we have partnered with the StreetEasy Research team. You will be working with a **.csv** file that contains a sample of 5,000 rentals listings in `Manhattan`, `Brooklyn`, and `Queens`, active on StreetEasy in June 2016.

It has the following columns:

Headers | Description |
--- | --- |
`rental_id` | rental ID
`building_id` | building ID
`rent` | price of rent ($)
`bedrooms` | number of bedrooms
`bathrooms` | number of bathrooms
`size_sqft` | size in square feet
`min_to_subway` | distance form subway station in minutes
`floor` | floor number
`building_age_yrs` | building's age in years
`no_fee` | does it have a broker fee? (0 for fee, 1 for no fee)
`has_roofdeck` | does it have a roof deck? (o for no, 1 for yes)
`has_washer_dryer` | does it have washer/dryer in unit (0/1
`has_doorman` | does it have a doorman? (0/1)
`has_elevator` | does it have an elevator? (0/1)
`has_dishwasher` | does it have a dishwasher? (0/1)
`has_patio` | does it have a patio? (0/1)
`has_gym` | does the building have a gym?  (0/1)
`neighborhood` | neighborhood (ex: Greenpoint)
`submarket` | submarket (ex: North Brooklyn)
`borough` | borough (ex: Brooklyn)

---

Thank you StreetEasy for this partnership and especially:

- [Grant Long](https://streeteasy.com/blog/author/grantlong/), Sr. Economist, StreetEasy
- [Lauren Riefflin](https://streeteasy.com/blog/author/lauren/), Sr. Marketing Manager, StreetEasy
- Philipp Kats, Data Scientist, StreetEasy
- Simon Rimmele, Data Scientist, StreetEasy
- Nancy Wu, Economic Data Analyst, Street Easy

Radu's code: 

Rent Prediction and Feature Analysis for Manhattan Apartments
This Python project explores the relationship between various apartment features and rent prices in Manhattan. The goal is to build a linear regression model to predict apartment rent based on multiple features, as well as to perform visualization to understand how different discrete and continuous features impact rent prices.

Requirements
To run this code, you'll need the following libraries:

matplotlib (for plotting)
numpy (for numerical operations)
pandas (for data manipulation)
seaborn (for data visualization)
scikit-learn (for machine learning and splitting the dataset)
scipy (for statistical testing)
You can install them using pip:

pip install matplotlib numpy pandas seaborn scikit-learn scipy
Project Overview
Data Preparation: The dataset manhattan.csv contains apartment listings in Manhattan, including features like the number of bedrooms, bathrooms, size in square feet, distance to the subway, building age, and other amenities (e.g., has a dishwasher, gym, etc.).

Data Splitting: The data is split into two sets:

Training Set (80%): Used to train the linear regression model.
Test Set (20%): Used to evaluate the model's performance.
Linear Regression Model: A multiple linear regression model (LinearRegression) is trained to predict the apartment's rent based on the input features. The performance of the model is evaluated using the R-squared score on both the training and test sets.

Visualization: Various visualizations are created to explore the relationship between apartment features and rent. These include:

Scatter plot of actual vs. predicted rent: To visualize how well the model's predictions align with the actual rent prices.
Boxplots for discrete features: For features with a low number of unique values, boxplots are created to show the distribution of rent within each category.
Scatter plots for continuous features: For features with many unique values, scatter plots are used to visualize how rent changes with each feature.
Code Breakdown
Data Loading and Preprocessing:

The dataset is loaded from a CSV file (manhattan.csv).
The independent variables (x) include various apartment features, and the dependent variable (y) is the rent.
python
streeteasy = pd.read_csv("manhattan.csv")
df = pd.DataFrame(streeteasy)
Feature Selection:

The features are divided into discrete and continuous categories based on the number of unique values they contain.
Discrete features have 10 or fewer unique values, while continuous features have more than 10 unique values.
python
discrete_features = [col for col in x.columns if x[col].nunique() <= discrete_threshold]
continuous_features = [col for col in x.columns if x[col].nunique() > discrete_threshold]
Model Training:

A multiple linear regression model is trained using the training set (x_train and y_train).
The model is then evaluated using the R-squared score (mlr.score), both on the training and test data (x_test and y_test).
python
mlr = LinearRegression()
mlr.fit(x_train, y_train)
r = mlr.score(x_train, y_train)
a = mlr.score(x_test, y_test)
Visualization:

A scatter plot is created to visualize the predicted vs. actual rent values.
Boxplots are generated for discrete features, comparing the distribution of rent for each category within those features.
Scatter plots are created for continuous features, showing how rent varies with each feature.
Example Outputs
Scatter Plot of Actual vs. Predicted Rent:

The scatter plot will show the correlation between the actual rent prices (y_test) and the predicted rent prices (y_predict). Points closer to the line y = x indicate better model predictions.
Boxplots:

For features like has_roofdeck, has_washer_dryer, and other binary features, the boxplots will visualize how rent distribution differs between apartments with and without the respective feature.
Scatter Plots:

For features like bedrooms, bathrooms, or size_sqft, scatter plots will show how rent changes as these features vary.
Key Takeaways
Linear Regression Model: The linear regression model is used to predict rent based on a set of features. The model's performance can be evaluated using the R-squared score.
Feature Analysis: Discrete and continuous features are analyzed separately, with appropriate visualizations to understand their relationship with rent.
Future Improvements
Experiment with other machine learning models (e.g., Random Forest, Gradient Boosting) to improve prediction accuracy.
Perform feature engineering to extract additional useful information from the dataset (e.g., combining features like size_sqft and bedrooms into a new feature representing rent per bedroom).


