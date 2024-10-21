import numpy as np
import pandas as pd

from types import MethodType
from typing import Optional
from warnings import warn

from numpy import ndarray
from scipy.stats import norm
from functools import partial
from itertools import combinations, zip_longest
from multiprocessing import Pool, cpu_count

from VBSA.VBSA.utils import extract_group_names
from VBSA.VBSA.plotting.results import ResultDict
CONST_RESULT_MSG = (
    "The input values are constant, "
    "therefore the Sobol indices cannot be calculated."
)



class SobolAnalyzer:
    def __init__(self,
                 problem: dict,
                 Y: ndarray,
                 calc_second_order=True,
                 num_resamples=1000,
                 conf_level=0.95,
                 parallel: Optional[bool] = None,
                 n_processors: Optional[int] = None,
                 keep_resamples: Optional[bool] = False,
                 print_to_console: Optional[bool] = False,
                 seed: Optional[int] = None):
        """
        Sobol Analyzer class to calculate the Sobol indices for a given model output values.
        :param problem: dict: problem dictionary for the model to be analyzed
        :param Y: ndarray: model output values to be analyzed for Sobol indices
        :param calc_second_order: bool: calculate second order indices or not
        :param num_resamples: int: number of resamples to perform on the model output values
        :param conf_level: float: confidence level of the Sobol indices
        :param parallel: bool: parallel flag to determine if the analysis is performed in parallel or not
        :param n_processors: int: number of processors to use for parallel analysis
        :param keep_resamples: bool: keep resamples or not
        :param print_to_console: bool: print the results to the console
        :param seed: int: seed value for the random number generator
        """
        self.problem = problem
        self.Y = Y
        self.calc_second_order = calc_second_order
        self.num_resamples = num_resamples
        self.conf_level = conf_level
        self.parallel = parallel
        self.n_processors = n_processors
        self.keep_resamples = keep_resamples
        self.print_to_console = print_to_console
        self.seed = seed
        self.rng = self.initialize_rng(seed)
        self.D, self.N = self.validate_inputs(problem, Y, calc_second_order, conf_level)
        self.Y = self.normalize_outputs(Y)
        self.A, self.B, self.AB, self.BA = self.separate_output_values(self.Y, self.D, self.N, calc_second_order)
        self.r = self.rng(self.N, size=(self.N, num_resamples))
        self.Z = norm.ppf(0.5 + conf_level / 2)


    def initialize_rng(self, seed: int) -> callable:
        """
        Initialize the random number generator.
        :param seed: int: seed value
        :return: callable: random number generator
        """
        return np.random.default_rng(seed).integers if seed else np.random.randint

    def validate_inputs(self, problem: dict, Y: ndarray, calc_second_order: bool, conf_level: int) -> tuple[int, int]:
        """
        Validate the inputs to the SobolAnalyzer class.
        :param problem: dict: problem dictionary
        :param Y: ndarray: model output values
        :param calc_second_order: bool: calculate second order indices
        :param conf_level: int: confidence level
        """
        if not 0 < conf_level < 1:
            raise ValueError("Confidence level must be between 0-1.")
        _, self.D = extract_group_names(problem)
        self.N = self.calculate_sample_size(self.D, calc_second_order, Y)
        return self.D, self.N

    def calculate_sample_size(self, D: int, calc_second_order: bool, Y: ndarray) -> int:
        """
        Calculate the sample size based on the number of parameters and whether second order indices are calculated.
        :param D: int: number of parameters
        :param calc_second_order: bool: calculate second order indices
        :param Y: ndarray: model output values
        :return: int: number of samples
        """
        if calc_second_order and Y.size % (2 * D + 2) == 0:
            return int(Y.size / (2 * D + 2))
        elif not calc_second_order and Y.size % (D + 2) == 0:
            return int(Y.size / (D + 2))
        else:
            raise ValueError("Incorrect number of samples in model output file. "
                             "Please ensure the number of samples is a multiple of D + 2 or 2D + 2.")

    def normalize_outputs(self, Y) -> tuple[ndarray, ndarray]:
        """
        Rescale the output values to have a mean of 0 and standard deviation of 1.
        :param Y: ndarray: model output values
        :return: tuple[ndarray, ndarray]: normalized output values
        """
        return (Y - Y.mean()) / Y.std()

    def analyze(self) -> ResultDict:
        """
        Analyze the model output values to calculate the Sobol indices.
        :return: ResultDict: dictionary of Sobol indices
        """
        kwargs = {
            'Z': self.Z,
            'A': self.A,
            'AB': self.AB,
            'BA': self.BA,
            'B': self.B,
            'r': self.r,
            'num_resamples': self.num_resamples,
            'keep_resamples': self.keep_resamples
        }

        if self.parallel:
            S = self.parallel_analysis(**kwargs)
        else:
            S = self.sequential_analysis(**kwargs)

        S.problem = self.problem
        S.to_df = MethodType(self.to_df, S)

        if self.print_to_console:
            self.print_results(S)

        return S

    def separate_output_values(self, Y: ndarray, D: int, N: int, calc_second_order: bool) -> tuple[ndarray, ndarray, ndarray, ndarray | None]:
        # Reshape Y to separate the values
        step = 2 * D + 2 if calc_second_order else D + 2
        Y_reshaped = Y.reshape(N, step)

        # Extract A and B
        self.A = Y_reshaped[:, 0]
        self.B = Y_reshaped[:, -1]

        # Extract AB
        self.AB = Y_reshaped[:, 1: D + 1]

        # Extract BA if second order indices are calculated
        self.BA = Y_reshaped[:, D + 1:2 * D + 1] if self.calc_second_order else None

        return self.A, self.B, self.AB, self.BA

    def sequential_analysis(self, **kwargs) -> ResultDict:
        """
        Perform the Sobol analysis sequentially. this function is called when the parallel flag is set to False.
        calculate the first and total order indices for each parameter. If the second order indices are calculated,
        also calculate the second order indices.
        :param kwargs: dict: keyword arguments
        :return: ResultDict: dictionary of Sobol indices
        """
        S = self.create_Si_dict(self.D, kwargs.get('num_resamples'), kwargs.get('keep_resamples'), self.calc_second_order)
        for j in range(self.D):
            S["S1"][j], S["S1_conf"][j] = self.calculate_first_order(kwargs.get('A'), kwargs.get('AB')[:, j], kwargs.get('B'), kwargs.get('r'), kwargs.get('Z'), kwargs.get('keep_resamples'), S, j)
            S["ST"][j], S["ST_conf"][j] = self.calculate_total_order(kwargs.get('A'), kwargs.get('AB')[:, j], kwargs.get('B'), kwargs.get('r'), kwargs.get('Z'), kwargs.get('keep_resamples'), S, j)
        if self.calc_second_order:
            self.calculate_second_order(self.D, kwargs.get('A'), kwargs.get('AB'), kwargs.get('BA'), kwargs.get('B'), kwargs.get('r'), kwargs.get('Z'), S)
        return S

    def parallel_analysis(self, **kwargs) -> ResultDict:
        """
        Perform the Sobol analysis in parallel. This function is called when the parallel flag is set to True.
        Calculate the first and total order indices for each parameter. If the second order indices are calculated,
        also calculate the second order indices.
        :param kwargs:
        :return: ResultDict: dictionary of Sobol indices
        """
        tasks, n_processors = self.create_task_list(self.D, self.calc_second_order, self.n_processors)
        func = partial(self.sobol_parallel, kwargs.get('Z'), kwargs.get('A'), kwargs.get('AB'), kwargs.get('BA'), kwargs.get('B'), kwargs.get('r'))
        with Pool(n_processors) as pool:
            S_list = pool.map(func, tasks)
        return self.Si_list_to_dict(S_list, self.D, kwargs.get('num_resamples'), kwargs.get('keep_resamples'), self.calc_second_order)

    def calculate_first_order(self, A: ndarray, AB_j: ndarray, B:ndarray, r: initialize_rng, Z: int, keep_resamples: bool, S: ResultDict, j: int) -> tuple[ndarray, float]:
        """
        Calculate the first order indices for each parameter. If keep_resamples is True, store the resamples in the S dictionary.
        :param A: ndarray: a matrix of model output values
        :param AB_j: ndarray: a matrix of model output values
        :param B: ndarray: a matrix of model output values
        :param r: initialize_rng: random number generator
        :param Z: int: confidence level of the Sobol indices
        :param keep_resamples: bool: keep resamples or not
        :param S: ResultDict: dictionary of Sobol indices
        :param j: int: index of the parameter
        :return: tuple[ndarray, float]: first order indices and confidence interval
        """
        S1 = self.first_order(A, AB_j, B)
        S1_conf_j = self.first_order(A[r], AB_j[r], B[r])
        if keep_resamples:
            S["S1_conf_all"][:, j] = S1_conf_j
        var_diff = np.ptp(np.r_[A[r], B[r]])
        S1_conf = Z * S1_conf_j.std(ddof=1) if var_diff != 0.0 else 0.0
        return S1, S1_conf

    def calculate_total_order(self, A: ndarray, AB_j: ndarray, B: ndarray, r: initialize_rng, Z: int, keep_resamples: bool, S: ndarray, j: int) -> tuple[ndarray, float | ndarray]:
        """
        Calculates the total order indices for each parameter. If keep_resamples is True, store the resamples in the S dictionary.
        total order is calculated as the variance of the model output values for each parameter. This is used to determine the
        importance of each parameter in the model.
        :param A: ndarray: a matrix of model output values
        :param AB_j: ndarray: a matrix of model output values
        :param B: ndarray: a matrix of model output values
        :param r: initialize_rng: random number generator
        :param Z: int: confidence level of the Sobol indices
        :param keep_resamples: bool: keep resamples or not
        :param S: ndarray: dictionary of Sobol indices
        :param j: int: index of the parameter
        :return: tuple[ndarray, float | ndarray]: total order indices and confidence interval
        """
        ST = self.total_order(A, AB_j, B)
        ST_conf_j = self.total_order(A[r], AB_j[r], B[r])
        if keep_resamples:
            S["ST_conf_all"][:, j] = ST_conf_j
        var_diff = np.ptp(np.r_[A[r], B[r]])
        ST_conf = Z * ST_conf_j.std(ddof=1) if var_diff != 0.0 else 0.0
        return ST, ST_conf

    def calculate_second_order(self, D, A, AB, BA, B, r, Z, S) -> None:
        for j in range(D):
            for k in range(j + 1, D):
                S["S2"][j, k] = self.second_order(A, B, AB[:, j], AB[:, k], BA[:, j])
                S["S2_conf"][j, k] = Z * self.second_order(A[r], B[r], AB[r, j], AB[r, k],
                                                           BA[r, j]).std(ddof=1)

    @staticmethod
    def print_results(S) -> None:
        res = S.to_df()
        for df in res:
            print(df)

    def first_order(self, A: ndarray, AB: ndarray, B: ndarray) -> ndarray:
        y = np.r_[A, B]
        if np.ptp(y) == 0:
            warn(CONST_RESULT_MSG)
            return np.array([0.0])
        return np.mean(B * (AB - A), axis=0) / np.var(y, axis=0)

    def total_order(self, A, AB, B) -> ndarray:
        try:
            y = np.r_[A, B]
            if np.ptp(y) == 0:
                warn(CONST_RESULT_MSG)
                return np.array([0.0])
            return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(y, axis=0)
        except ValueError as e:
            raise f"Error in total_order: {e}"

    def second_order(self,A: ndarray, B: ndarray, ABj: ndarray, ABk: ndarray, BAj: ndarray) -> ndarray:
        y = np.r_[A, B]
        if np.ptp(y) == 0:
            warn(CONST_RESULT_MSG)
            return np.array([0.0])
        Vjk = np.mean(BAj * ABk - A * B, axis=0) / np.var(y, axis=0)
        Sj = self.first_order(A, ABj, B)
        Sk = self.first_order(A, ABk, B)
        return Vjk - Sj - Sk

    def create_Si_dict(self, D, num_resamples, keep_resamples, calc_second_order) -> ResultDict:
        S = ResultDict((k, np.zeros(D)) for k in ("S1", "S1_conf", "ST", "ST_conf"))
        if keep_resamples:
            S["S1_conf_all"] = np.zeros((num_resamples, D))
            S["ST_conf_all"] = np.zeros((num_resamples, D))
        if calc_second_order:
            S["S2"] = np.full((D, D), np.nan)
            S["S2_conf"] = np.full((D, D), np.nan)
        return S

    def sobol_parallel(self, Z: int, A: ndarray, AB: ndarray, BA: ndarray, B: ndarray, r: initialize_rng, tasks: list) -> list:
        """
        Parallel function to calculate the Sobol indices. This function is called when the parallel flag is set to True.
        The function calculates the Sobol indices for each parameter in parallel by splitting the tasks into n_processors.
        Then, the results are combined into a single list.
        :param Z: int: confidence level of the Sobol indices
        :param A: ndarray: a matrix of model output values
        :param AB: ndarray: a matrix of model output values
        :param BA: ndarray: a matrix of model output values
        :param B: ndarray: a matrix of model output values
        :param r: initialize_rng: random number generator
        :param tasks: list: list of tasks to be performed
        :return: list: list of Sobol indices
        """
        sobol_indices = []
        task_map = {
            "S1": lambda A, AB, B, Z, r, j, k: self.first_order(A, AB[:, j], B),
            "S1_conf": lambda A, AB, B, Z, r, j, k: Z * self.first_order(A[r], AB[r, j], B[r]).std(ddof=1),
            "ST": lambda A, AB, B, Z, r, j, k: self.total_order(A, AB[:, j], B),
            "ST_conf": lambda A, AB, B, Z, r, j, k: Z * self.total_order(A[r], AB[r, j], B[r]).std(ddof=1),
            "S2": lambda A, AB, B, Z, r, j, k: self.second_order(A, AB[:, j], AB[:, k], BA[:, j], B),
            "S2_conf": lambda A, AB, B, Z, r, j, k: Z * self.second_order(A[r], AB[r, j], AB[r, k], BA[r, j], B[r]).std(
                ddof=1),
        }

        for d, j, k in tasks:
            s = task_map[d](A, AB, B, Z, r, j, k)
            sobol_indices.append([d, j, k, s])
        return sobol_indices

    def create_task_list(self, D: int, calc_second_order: bool, n_processors: int) -> tuple[list, int]:
        """
        Creates a list of tasks to be performed in parallel. The tasks are split into n_processors.
        :param D: int: number of parameters
        :param calc_second_order: bool: calculate second order indices
        :param n_processors: int: number of processors
        :return: tuple[list, int]: list of tasks and number of processors
        """
        tasks_first_order = [
            [d, j, None] for j in range(D) for d in ("S1", "S1_conf", "ST", "ST_conf")
        ]
        tasks_second_order = []
        if calc_second_order:
            tasks_second_order = [
                [d, j, k]
                for j in range(D)
                for k in range(j + 1, D)
                for d in ("S2", "S2_conf")
            ]
        if n_processors is None:
            n_processors = min(
                cpu_count(), len(tasks_first_order) + len(tasks_second_order)
            )
        if not calc_second_order:
            tasks = np.array_split(tasks_first_order, n_processors)
        else:
            tasks = np.array_split(
                [
                    v
                    for v in sum(
                        zip_longest(tasks_first_order[::-1], tasks_second_order), ()
                    )
                    if v is not None
                ],
                n_processors,
            )
        return tasks, n_processors

    def Si_list_to_dict(self, S_list: list, D: int, num_resamples: int, keep_resamples: bool, calc_second_order: bool) -> ResultDict:
        S = self.create_Si_dict(D, num_resamples, keep_resamples, calc_second_order)
        L = []
        for list in S_list:
            L += list
        for s in L:
            if s[2] is None:
                S[s[0]][s[1]] = s[3]
            else:
                S[s[0]][s[1], s[2]] = s[3]
        return S

    def to_df(self):
        total, first, (idx, second) = self.Si_to_pandas_dict()
        names, _ = extract_group_names(self.problem)
        ret = [pd.DataFrame(total, index=names), pd.DataFrame(first, index=names)]
        if second:
            ret += [pd.DataFrame(second, index=idx)]
        return ret

    @staticmethod
    def Si_to_pandas_dict(S_dict):
        total_order = {"ST": S_dict["ST"], "ST_conf": S_dict["ST_conf"]}
        first_order = {"S1": S_dict["S1"], "S1_conf": S_dict["S1_conf"]}
        idx = None
        second_order = None
        if "S2" in S_dict:
            names, _ = extract_group_names(S_dict.problem)
            if len(names) > 2:
                idx = list(combinations(names, 2))
            else:
                idx = (names,)
            second_order = {
                "S2": [S_dict["S2"][names.index(i[0]), names.index(i[1])] for i in idx],
                "S2_conf": [
                    S_dict["S2_conf"][names.index(i[0]), names.index(i[1])] for i in idx
                ],
            }
        return total_order, first_order, (idx, second_order)

# TODO: test the analyze function to ensure it returns the correct values as SALib
