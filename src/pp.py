import pandas as pd
import numpy as np

from typing import Tuple
from typing import Callable
from sklearn.base import BaseEstimator

def pair_permutation_test(model1_pred: np.array, 
                          model2_pred: np.array, 
                          ground_truth:np.array, 
                          n_permutations:10000,
                          metric: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[float, float, float, float, float]:
    """
        Calculates Statistical Significance level p-value for the given pair
    """

    # Calculate percent error of the models
    model1_scores      = metric(ground_truth, model1_pred)
    model2_scores      = metric(ground_truth, model2_pred)
    
    # Calcualte Observed Difference
    observed_statistic = np.abs(model1_scores - model2_scores)
    
    # Number of permutations
    permutation_statistics = np.zeros(n_permutations)
    
    # Permutation process
    for i in range(n_permutations):
        model1_pred_temp = model1_pred.copy()
        model2_pred_temp = model2_pred.copy()

        # Shuffle the prediction values
        random_indexs                    = np.random.random(size=model1_pred.shape[0]) < 0.5
        model1_pred_temp[random_indexs]  = model2_pred[random_indexs]
        model2_pred_temp[random_indexs]  = model1_pred[random_indexs]

        permutation_statistics[i]        = np.abs(metric(ground_truth, model1_pred_temp) - metric(ground_truth, model2_pred_temp))

    # Calculate p-value
    p_value = np.mean(permutation_statistics >= observed_statistic)

    return observed_statistic, p_value, permutation_statistics.mean(), permutation_statistics.std()


def find_paired_permutation_test(dataset:Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                                 model1_:BaseEstimator,
                                 model2_:BaseEstimator,
                                 model1_name:str,
                                 model2_name:str) -> pd.DataFrame:
    """
        Calculates Statistical Significance level p-value for all the given pairs
    """

    df      = pd.DataFrame(columns=['Model Comparison', 'Observed Diff', 'Diff mean', 'Diff std', 'p value'])
    (X_train, X_test, y_train, y_test) = dataset
    
    # Load Models
    model1  = model1_.copy()
    model2  = model2_.copy()

    # Fit model with the best features
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # Predict concentration on testing dataset
    model1_pred = model1.predict(X_test)
    model2_pred = model2.predict(X_test)

    # Peform permutation test
    _, observed_diff, p_value, diff_mean, diff_std = pair_permutation_test(model1_pred, model2_pred, y_test)

    temp = pd.DataFrame.from_dict({'Model Comparison':[f"{model1_name}--------- {model2_name}"], \
                                'Observed Diff':[observed_diff], \
                                'Diff mean':diff_mean, 'Diff std':diff_std,'p value':[p_value]})
    
    df   = pd.concat([df, temp], ignore_index=True)

    return df