## Build a machine learning algorithm to predict activity quality from activity monitors. 
You can view the complete analysis and results [here](https://msholk.github.io/ml_final_project/ML_Final_Predictions01.html).

### Data Source:

I am using the Weight Lifting Exercises Dataset from HAR (Human Activity Recognition) data, which contains sensor data from wearable devices attached to subjects performing exercises.

### Prediction Goal:

The model’s objective is to predict the type of the performed activity, represented by the target variable **classe**.

### Feature Engineering & Data Preparation:

I clean the dataset by:
 - Removing irrelevant or redundant features (timestamps, identifiers).
 - Handling missing values and near-zero variance features.
 - Performing feature selection based on correlation analysis.

### Machine Learning Algorithm

I trained multiple machine learning models to predict activity type from sensor data:

- **Random Forest** (`ranger`)
- **Regularized Generalized Linear Model** (`glmnet`)
- **Extreme Gradient Boosting** (`xgbTree`)

To enhance predictive performance, I applied **model stacking** using **`xgbTree`** as the meta-learner, which combines the predictions from the base models to improve overall accuracy.


### Model Evaluation:

I have evaluated the model's performance using a confusion matrix and calculated key metrics such as accuracy.

### Out-of-Sample Error Estimation

The **out-of-sample error** estimates how well the model is expected to perform on new, unseen data. A low out-of-sample error indicates that the model generalizes well and is not overfitting the training data.

Given the complexity of the sensor data and the nature of the classification task, we expect the out-of-sample error to be relatively low (ideally below 5%) for a well-tuned model.

The received value  **0.34%** suggests that the model is not overfitting and is likely to perform well on new data.

In this analysis, we use **cross-validation** combined with a confusion matrix to estimate **the out-of-sample error**. 



### Parallel Processing:

I have implemented parallel processing, which helps speed up model training for large datasets.
