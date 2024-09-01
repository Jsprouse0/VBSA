## Variance Based Sensitivity Analysis (NetPyNE)
#
This project demonstrates the application of Variance-Based Sensitivity Analysis (VBSA) on various functions using Sobol's sensitivity 
indices. The code is intended to serve as a foundation for more advanced VBSA optimizations across different cells, with continuous improvements and additions planned.
# Overview
The current implementation focuses on:
- Rosenbrock Function: The Rosenbrock function is used as the target model, which is a well-known test problem for optimization algorithms.
- Variance-based Sensitivity Analysis (VBSA): The VBSA is performed using Sobol's method to analyze the sensitivity of the model's output with respect to its input variables
- Sobol Sampling: The input parameters are generated using Sobol's sampling technique to ensure a comprehensive exploration of the input space.
- Sensitivity Analysis: The first-order and total-order Sobol indices are computed to measure the sensitivity of each input variable on the output.
# How to Use
1. Install Dependencies: SALib and numpy are the necessary Python Packages to run this for now.
2. Run the VBSA Analysis
3. Review Results: Sensitivity Indices are printed to the console and optionally saved to the results folder for further analysis.

# Citations:
- Iwanaga, T., Usher, W., & Herman, J. (2022). Toward SALib 2.0: Advancing the accessibility and 
    interpretability of global sensitivity analyses. Socio-Environmental Systems Modelling, 4(1), 1–15. 
    https://doi.org/10.18174/sesmo.18155