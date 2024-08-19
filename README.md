This repository contains code to perform paired permutation tests in different machine-learning models with a notebook to show how to use the functions. You can customize the code for your needs.<br>
Please find the detailed article here <a href='https://medium.com/@sangambuddhacharya/paired-permutation-test-in-machine-learning-787460731e68'>https://medium.com/@sangambuddhacharya/paired-permutation-test-in-machine-learning-787460731e68 </a>

## Install Dependencies
```bash
  pip install -r requirements.txt
```

To see how to use the function, open the Run ```test_pp.ipynb``` notebook. In the example, I used a simple insurance dataset with 6 features and tested it with two models: Linear Regression and Decision Tree. Feel free to use any model you prefer.


## Function Description
```find_paired_permutation_test``` function is in src/pp.py.
You can directly compute paired permutation tests using this function.

```python
def find_paired_permutation_test(dataset:Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                                 model1:BaseEstimator,
                                 model2:BaseEstimator,
                                 model1_name:str,
                                 model2_name:str, 
                                 metric: Callable[[np.ndarray, np.ndarray], float],
                                 n_permutations=10000) -> pd.DataFrame:

# model1: first model to compare (scikit-model) (e.g. LinearRegression())
# model2: second model to compare (scikit-model) (e.g. SVR())
# model1_name: Name of the first model (e.g Linear)
# model1_name: Name of the second model (e.g SVM)
# metric:  Metric used to evaluate the performance of the model (eg. Accuracy, MSE, R2, ...)
```

If you want to customize paired permutation test for you code please use the following function.
```pair_permutation_test``` function is in src/pp.py
```python
def pair_permutation_test(model1_pred: np.array, 
                          model2_pred: np.array, 
                          ground_truth:np.array, 
                          n_permutations:10000,
                          metric: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[float, float, float, float, float]:


# model1_pred    : It is the first model's prediction output (numpy array).
# model2_pred    : It is the second model's prediction output (numpy array).
# ground_truth   : It is the target value of the testing dataset (numpy array).
# n_permutations : Number of samples to generate the distribution
# metric         : Metric used to evaluate the performance of the model (eg. Accuracy, MSE, R2, ...)
```

## Output
```
Model Comparison	 Linear	      SVM	 Observed_Diff    Diff_mean	 Diff_std	  p_value
0	Linear | SVM	  0.805552	0.87408	     0.051856	     0.098262	  0.07112	   0.6843

Linear (Model1):  metric score on the testing dataset
SVM (Model2):     metric score on the testing dataset
Observed_Diff:    the absolute difference between scores of the two model
Diff_mean:        mean of the distribution of the absolute difference between the model score after shuffling the model for N iteration
Diff_std:         standard deviation of the distribution of the absolute difference between the model score after shuffling the model for N iteration
p_value:          a statistical measurement used to validate a null hypothesis. (The hypothesis fails if p<0.005, indicating a 95% probability that the observed difference in performance is not due to randomness.)
```



