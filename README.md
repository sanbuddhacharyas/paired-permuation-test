This repository contains code to perform paired permutation tests in different machine-learning models with a notebook to show how to use the functions. You can customize the code for your needs.


## Install Dependencies
```bash
  pip install -r requirements.txt
```

To see how to use the function run ```Run test_pp.ipynb``` notebook.<be>

## Function Description
```pair_permutation_test``` function is in src/pp.py
```python
def pair_permutation_test(model1_pred: np.array, 
                          model2_pred: np.array, 
                          ground_truth:np.array, 
                          n_permutations:10000,
                          metric: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[float, float, float, float, float]:

```

