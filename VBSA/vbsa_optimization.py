from Rosenbrock.rosenbrock import rosenbrock_v0
import numpy as np

class VBSAOptimization:
    def __init__(self, init_file,
                 problem: dict,
                 param_values: np.ndarray,
                 model_results: np.ndarray,
                 sobol_indices: dict):
        self.init_file = init_file
        self.problem = problem
        self.param_values = param_values
        self.model_results = model_results
        self.sobol_indices = sobol_indices

    def
