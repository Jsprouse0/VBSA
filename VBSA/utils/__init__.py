import pandas as pd
from typing import Dict, Tuple, List

def extract_group_names(problem: Dict) -> Tuple[List[str], int]:
    """Extract unique group names from the problem definition

    If groups are not defined, use the variable names as groups.

    :param  problem : dict
        The problem definition containing 'names' and optionally 'groups'.
    :return: tuple : (list of unique group names, number of groups) -> immutable
    """
    groups = problem.get("groups", problem["names"])
    unique_groups = list(dict.fromkeys(groups))
    number_of_groups = len(unique_groups)

    return unique_groups, number_of_groups


def read_param_file(file_path: str) -> Dict:
    """Read the parameter file and return the problem definition

    :param  file_path : str
        The path to the parameter file.
    :return: dict
        The problem definition.
    """
    problem = pd.read_json(file_path, orient="records").to_dict(orient="records")[0]

    return problem