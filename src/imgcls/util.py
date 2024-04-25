import polars as pl


__all__ = ['printdf']

def printdf(df: pl.DataFrame,
            nrows: int | None = None,
            ncols: int | None = None,
            tbl_width_chars: int = 500,
            do_print: bool = True) -> str:
    """
    print dataframe with given row numbers (polars)
    if isinstance pandas dataframe, print all.

    :param df: polars or pandas dataframe
    :param nrows: number of rows (applicable in polars case)
    :param ncols: number of columns
    :param tbl_width_chars: table width for showing
    :param do_print: do print otherwise, only return the str
    :return:
    """

    with pl.Config(tbl_width_chars=tbl_width_chars) as cfg:
        rows = df.shape[0] if nrows is None else nrows
        cols = df.shape[1] if ncols is None else ncols
        cfg.set_tbl_rows(rows)
        cfg.set_tbl_cols(cols)

        if do_print:
            print(df)

        return df.__repr__()
