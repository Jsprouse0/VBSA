from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from VBSA.VBSA.Rosenbrock.rosenbrock import rosenbrock_v0
import numpy as np

# Define the problem for the VBSA
problem = {
    'num_vars': 4,  # Number of variables (x0, x1, x2, x3)
    'names': ['x0', 'x1', 'x2', 'x3'],
    'bounds': [[-5, 5], [-5, 5], [-5, 5], [-5, 5]]  # Bounds for each variable
}

# Generate samples using Sobol's sampling scheme (Saltelli is depricated)
param_values = sobol_sample.sample(problem, 1024)  # use a number of samples (N) that is a power of 2. You can adjust the sample size accordingly
print(param_values[0])

# Run the model (Rosenbrock function) for each set of samples
model = np.array([rosenbrock_v0(*params) for params in param_values])

# Perform the VBSA (Sobol Sensitivity Analysis)
Si = sobol_analyze.analyze(problem, model)

# Print sensitivity indices
print("First-order Sobol indices:", Si['S1'])
print("Total-order Sobol indices:", Si['ST'])

# Optionally, save the results to a file in the results/ directory
with open('../results/sensitivity_indices.txt', 'w') as f:
    f.write(f"First-order Sobol indices: {Si['S1']}\n")
    f.write(f"Total-order Sobol indices: {Si['ST']}\n")