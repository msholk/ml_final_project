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

I trained multiple machine learning models to predict activity type from sensor data:
 - Random Forest (ranger)
 - Regularized Generalized Linear Model (glmnet)
 - Extreme Gradient Boosting (xgbTree)
   
To enhance predictive performance, I applied model stacking using xgbTree as the meta-learner, which combines the predictions from the base models to improve overall accuracy.

### Model Evaluation:

We evaluated the model's performance using a confusion matrix and calculated key metrics such as accuracy.
The model also estimates the out-of-sample error using cross-validation.

### Parallel Processing:

We implemented parallel processing, which helps speed up model training for large datasets.
