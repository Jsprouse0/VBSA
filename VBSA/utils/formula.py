import numpy as np

import numpy as np

def calculate_threshold(data: np.ndarray, factor: float = 1.0) -> float:
    """
    Calculate a dynamic threshold based on the mean and standard deviation of the data.
    :param data: np.ndarray: Input data array
    :param factor:  float: Multiplicative factor to adjust the threshold (can increase to make more lenient or decrease to make more strict)
    :return: float: Calculated threshold
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    return mean + factor * std_dev

def check_predictions_shape(predictions, y,):
    assert predictions.shape == y.shape, f"Expected predictions to have shape {y.shape} but got {predictions.shape}"

def mean_squared_error(mse, threshold=0.1):
    assert mse > threshold, f"Mean Squared Error (MSE) is too low. Got: {mse}, Expected: > {threshold}"
    return mse

def mean_absolute_error(mae, threshold=0.5):
    assert mae > threshold, f"Mean Absolute Error (MAE) is too low. Got: {mae}, Expected: > {threshold}"
    return mae

def negative_log_likelihood(nll, threshold=10):
    assert nll < threshold, f"Negative Log Likelihood (NLL) is too high. Got: {nll}, Expected: < {threshold}"
    return nll

def sum_of_squares(ss, threshold=1.0):
    assert ss < threshold , f"Sum of Squares (SS) is too high. Got: {ss}, Expected: < {threshold}"
    return ss
