import pandas as pd
from VBSA.VBSA.analyze.sobol import Si_to_pandas_dict
from VBSA.VBSA.utils import extract_group_names

def to_df(problem):
    total, first, (idx, second) = Si_to_pandas_dict()
    names, _ = extract_group_names(problem)
    ret = [pd.DataFrame(total, index=names), pd.DataFrame(first, index=names)]
    if second:
        ret += [pd.DataFrame(second, index=idx)]
    return ret