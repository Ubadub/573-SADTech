#!/usr/bin/env python3.9

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

true_labels_array = np.array([2,2,1,1,0])
predicted_labels_array = np.array([2,1,1,0,0])

# true_labels_array = ["HIGHLY NEGATIVE","HIGHLY NEGATIVE","NEGATIVE","NEGATIVE","NEUTRAL"]
# predicted_labels_array = ["HIGHLY NEGATIVE","NEGATIVE","NEGATIVE","NEUTRAL","NEUTRAL"]

# works for both str and numerical vals
def ACC_calc(true_ar: np.ndarray, pred_ar: np.ndarray) -> np.ndarray:
    return accuracy_score(true_ar,pred_ar)

# works for both str and numerical vals
def F1_calc(true_ar: np.ndarray, pred_ar: np.ndarray) -> np.ndarray:
    return f1_score(true_ar,pred_ar,average=None)

# works just for numerical vals
def RMSE_calc(true_ar: np.ndarray, pred_ar: np.ndarray) -> np.float64:
    return - mean_squared_error(true_ar,pred_ar)

def main():
    acc = ACC_calc(true_labels_array, predicted_labels_array)
    print(f"Accuracy: " + str(acc))
    f1 = F1_calc(true_labels_array, predicted_labels_array)
    print(f"F1: " + str(f1))
    rmse = RMSE_calc(true_labels_array, predicted_labels_array)
    print(f"RMSE: " + str(rmse))

if __name__ == '__main__':
    main()
