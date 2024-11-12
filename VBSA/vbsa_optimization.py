import numpy as np

from VBSA.VBSA.analyze.sobol import SobolAnalyzer
from VBSA.VBSA.plotting.results import ResultDict
import VBSA.VBSA.utils.formula as formula

class VBSAOptimization:
    def __init__(self, problem: dict, num_samples: int, **kwargs):
        """
        Initialize the VBSAOptimization class.
        :param problem: Dictionary containing the problem definition.
                - num_vars (int): Number of input variables.
                - names (list): Names of the input variables.
                - bounds (np.ndarray): Bounds of the input variables.
                - num_outputs (int): Number of model outputs.
        :type problem: dict
        :param num_samples: Number of samples to generate for the sensitivity analysis.
        :param **kwargs Additional parameters for the SobolAnalyzer.
                - calc_second_order (bool, optional): Calculate second order indices or not.
                - num_resamples (int, optional): Number of resamples to perform on the model output values.
                - conf_level (float, optional): Confidence level of the Sobol indices.
                - parallel (bool, optional): Parallel flag to determine if the analysis is performed in parallel or not.
                - n_processors (int, optional): Number of processors to use for parallel analysis.
                - keep_resamples (bool, optional): Keep resamples or not.
                - print_to_console (bool, optional): Print the results to the console.
                - seed (int, optional): Seed value for the random number generator.
        """
        self.problem = problem
        self.num_samples = num_samples
        self.kwargs = kwargs
        self.Y = None

    def analyze(self):
        analyzer = SobolAnalyzer(problem=self.problem, num_samples=self.num_samples, **self.kwargs)
        print(f"Analyzing model with {self.num_samples} samples...")
        self.Y = analyzer.Y
        return analyzer.analyze()

    @staticmethod
    def initialize_params(problem: dict, method: str = "random") -> np.ndarray | None:
        """
        Initialize the parameters for the model optimization based on the specified method.
        :param problem: dict: Problem definition containing the number of variables and bounds.
        :param method: str: Initialization method for the parameters. (e.g., random, zero, custom)
        :return: np.ndarray: Initialized parameters for the model optimization.
        """
        if method == "random":
            return np.random.uniform(problem["bounds"][:, 0], problem["bounds"][:, 1], problem["num_vars"])
        elif method == "zero":
            return np.zeros(problem["num_vars"])
        elif method == "custom":
            return None
        else:
            raise ValueError(f"Initialization method '{method}' not supported. Choose from: random, zero, custom")

    @staticmethod
    def predict(X: np.ndarray, params: np.ndarray, method: str = "linear") -> np.ndarray:
        """
        Predict the model output based on the input parameters and feature matrix. \n
        :param X: np.ndarray: Input Features (each row corresponds to a sample, each column corresponds to a feature).
        :param params: np.ndarray: Input parameters for the model. (a vector of parameters-weights-for the model)
        :param method: str: Model prediction method. (e.g., linear, logistic, neural network)
        :return: np.ndarray: Model predictions based on the input parameters and feature matrix.
        """
        def custom_predict(X, params):
            # Custom prediction function based on the user-defined model
            pass

        prediction_methods = {
            "linear": lambda X, params: X @ params,
            "logistic": lambda X, params: 1 / (1 + np.exp(-X @ params)),
            "neural_network": lambda X, params: None,
            "random_forest": lambda X, params: None,
            "polynomial": lambda X, params: None,
            "custom": custom_predict if method == "custom" else None
        }

        if method in prediction_methods:
            return prediction_methods[method](X, params)
        else:
            raise ValueError(f"Prediction method '{method}' not supported. Choose from: {list(prediction_methods.keys())}")

    @staticmethod
    def objective_function(params: np.ndarray, X: np.ndarray, y: np.ndarray, objective: str="SS", custom_func=None, method: str="linear") -> float:
        """
        Objective function that quantifies the performance of the model based on the input parameters.
        This function should be defined based on the specific optimization goal (e.g., minimizing error, maximizing performance). \n
        - "MSE": Mean Squared Error (Commonly used in machine learning to measure the average squared difference between predicted and actual values).
        - "NLL": Negative Log-Likelihood (Used in probabilistic models to measure the likelihood of the model given the observed data).
        - "CPM": Custom Performance Metric (Custom metric defined based on the specific problem and model).
        - "SS": Sum of Squares (Often used in regression problems to minimize error).
        - "R2": Coefficient of Determination (Measures the proportion of variance in the dependent variable that is predictable from the independent variables).
        - "MAE": Mean Absolute Error (Commonly used in regression to measure the average absolute difference between predicted and actual values).
        - "RMSE": Root Mean Squared Error (Similar to MSE but takes the square root of the error).
        - "MAPE": Mean Absolute Percentage Error (Measures the average percentage difference between predicted and actual values).

        Custom Performance Metric Function: \n
        - The user can provide a custom performance metric function to evaluate the model performance.
        - The function should take the predicted values and actual values as input and return a scalar value.
        - If no custom function is provided, the default metric used is the Mean Absolute Error (MAE).
        - This function can be used for user specific performance metrics that are not covered by the default options.
        - Example: objective_function(params, X, y, objective="CPM", custom_func=custom_metric_func)

        :param params: np.ndarray: Input parameters for the model. (a vector of parameters-weights-for the model)
        :param X: np.ndarray: Input Features (each row corresponds to a sample, each column corresponds to a feature).
        :param y: np.ndarray: Actual target Values. (a vector of actual target values)
        :param objective: str: Optimization objective.
        :param custom_func: Optional custom performance metric function input by the user.
        :param method: str: Model prediction method. (e.g., linear, logistic, neural network) default is linear.
        :return float: Value of the specified performance metric
        """

        # Ensures that the number of elements in y matches the number of rows in X (i.e., 100 if np.random.rand(100, D))
        y = y[:X.shape[0]]

        # Example model prediction (linear model)
        predictions = VBSAOptimization.predict(X, params, method=method)

        try:
            formula.check_predictions_shape(predictions, y)
        except AssertionError:
            raise ValueError(f"Predictions shape {predictions.shape} does not match target shape {y.shape}")

        # Calculate performance metrics
        performance_metrics = {
            'MSE': np.mean((predictions - y) ** 2),
            'NLL': -np.sum(y * np.log(predictions) - predictions),
            'CPM': custom_func(predictions, y) if custom_func else np.mean(np.abs(predictions - y)),
            'SS': np.sum(np.square(y - np.mean(y))),
            'R2': 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)),
            'MAE': np.mean(np.abs(predictions - y)),
            'RMSE': np.sqrt(np.mean((predictions - y) ** 2)),
            'MAPE': np.mean(np.abs((y - predictions) / y)) * 100
        }

        if objective in performance_metrics:
            threshold = formula.calculate_threshold(predictions)
            if objective == 'MSE':
                formula.mean_squared_error(performance_metrics['MSE'], threshold)
            elif objective == 'SS':
                formula.sum_of_squares(performance_metrics['SS'], threshold)
            elif objective == 'NLL':
                formula.negative_log_likelihood(performance_metrics['NLL'], threshold)
            elif objective == 'MAE':
                formula.mean_absolute_error(performance_metrics['MAE'], threshold)
            return performance_metrics[objective]
        else:
            raise ValueError(f"Objective function '{objective}' not supported. Choose from: {list(performance_metrics.keys())}")


    def optimize_model(self, data: ResultDict):
        """
        Optimize the model based on the objective_function results.\n
        1.) Perform Sensitivity Analysis:
            - Calculate Sobol indices to identify the most influential parameters.
        2.) Define an Objective Function:
            - This function should represent the goal of your optimization (e.g., minimizing error, maximizing performance).
        3.) Select an Optimization Algorithm:
            - Choose an optimization algorithm that suits your problem (e.g., genetic algorithms, gradient-based methods, Bayesian Optimization).
        4.) Optimize Parameters:
            - Ues the optimization algorithm to adjust the most influential parameters identified in the sensitivity analysis.
        :param data: ResultDict: dictionary of Sobol indices
        """

        most_influential_param = np.argmax(results["ST"])
        print(f"Optimizing model based on the most influential parameter: {self.problem['names'][most_influential_param]}")



if __name__ == "__main__":
    # Example usage of the VBSAOptimization class
    # Define the problem by specifying the number of variables, their names, bounds, and number of outputs
    problem = {
        "num_vars": 2,
        "names": ["x1", "x2"],
        "bounds": np.array([[0, 1], [0, 1]]),
        "num_outputs": 1
    }

    # D is the number of input variables (in this case 2)
    D = problem["num_vars"]
    params = VBSAOptimization.initialize_params(problem=problem)

    X = np.random.rand(100, D)

    vbsa = VBSAOptimization(problem=problem, num_samples=100)
    results = vbsa.analyze()
    Y = vbsa.Y

    print("First Order Indices: %s" % results["S1"])
    print("Total Order Indices: %s\n" % results["ST"])

    # Objective function example: Mean Squared Error (MSE) or Sum of Squares (SS)
    print(f"Objective Function Value Mean Squared Error: {vbsa.objective_function(params, X, Y, objective='MSE')}")
    print(f"Objective Function Value Mean Absolute Error: {vbsa.objective_function(params, X, Y, objective='MAE')}")
    print(f"Objective Function Value Negative Log-Likelihood: {vbsa.objective_function(params, X, Y, objective='NLL', method = 'logistic')}")
    print(f"Objective Function Value Sum of Squares: {vbsa.objective_function(params, X, Y, objective='SS')}")

# Sampling: Sobol sequences: Use low discrepancy sequences to sample the design space and generating sample points.
# Sufficient Sample Size: Ensure the sample size is large enough to capture the variability of the model output.
# Normalization: Normalize input parameters and model outputs to improve the stability and accuracy of the analysis
# Implement efficient Computation: Use parallel processing and vectorized operations to speed up the optimization process.
# Validate and Verify: Cross-validation to verify the robustness of the sensitivity indices. Consistency checks with other sensitivity analysis methods.
# Interpretation: Interpret the sensitivity indices to identify the most influential parameters and their interactions.
# Document and Communicate Findings: Clearly document the methodology, results, and conclusions of the sensitivity analysis for communication and decision-making purposes.
# Visualization: Use visualizations such as bar charts, scatter plots, and sensitivity plots to present the sensitivity analysis results in an intuitive and informative way.

# Step 1: Get Sobol Indices (Done)
# ToDo Step 2: Define Objective Function (takes the parameters as input and returns a value that quantifies the performance of the model)
# ToDo Step 3: Select Optimization Algorithm (Gradient Descent, Genetic Algorithms, Bayesian Optimization, Nelder-Mead Method, BFGS: Broyden-Fletcher-Goldfarb-Shanno, CMA-ES: Covariance Matrix Adaptation Evolution Strategy)
# ToDO Step 3 (2): Implement Gradient Descent Algorithm
# ToDo Step 4: Optimize Parameters
# Step 5: Update Model Parameters
# Step 6: Validate and Verify the Optimized Model
# Step 7: Compare with Baseline Model
# Step 8: Document and Communicate Results