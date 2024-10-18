__all__ = ["plot"]

CONF_COLUMN = "_conf"

def plot(Si_df, ax=None):
    conf_cols = Si_df.columns.str.contains(CONF_COLUMN)

    confs = Si_df.loc[:, conf_cols]
    confs.columns = [c.replace(CONF_COLUMN, "") for c in confs.columns]

    Sis = Si_df.loc[:, ~conf_cols]

    ax = Sis.plot(kind="bar", yerr=confs, ax=ax)
    return ax