## Build a machine learning algorithm to predict activity quality from activity monitors. 

### Data Source:

We are using the Weight Lifting Exercises Dataset from HAR (Human Activity Recognition) data, which contains sensor data from wearable devices attached to subjects performing exercises.

### Prediction Goal:

The model’s objective is to predict the type of the performed activity, represented by the target variable **classe**.

### Feature Engineering & Data Preparation:

We clean the dataset by:
 - Removing irrelevant or redundant features (timestamps, identifiers).
 - Handling missing values and near-zero variance features.
 - Performing feature selection based on correlation analysis.

### Machine Learning Algorithm:

We employed a Random Forest model, which is a machine learning algorithm for classification tasks.
We used cross-validation (trainControl) to train the model robustly and prevent overfitting.

### Model Evaluation:

We evaluated the model's performance using a confusion matrix and calculated key metrics such as accuracy.
The model also estimates the out-of-sample error using cross-validation.

### Parallel Processing:

We implemented parallel processing, which helps speed up model training for large datasets.
