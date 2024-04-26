from datetime import datetime
from typing import Literal

import polars as pl
from colorama import Fore, Style

__all__ = ['printdf',
           'fprint']


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


def fprint(*msgs,
           vtype: Literal['info', 'io', 'warning', 'error', 'pass'] = 'info',
           timestamp: bool = True,
           **kwarg) -> None:
    """
    Formatting print with different colors based on verbose type

    :param msgs:
    :param vtype: verbose type
    :param timestamp:
    :return:
    """

    if vtype == 'error':
        prefix = '[ERROR]'
        color = 'red'
    elif vtype == 'warning':
        prefix = '[WARNING] '
        color = 'yellow'
    elif vtype == 'io':
        prefix = '[IO] '
        color = 'magenta'
    elif vtype == 'info':
        prefix = '[INFO]'
        color = 'cyan'
    elif vtype == 'pass':
        prefix = '[PASS]'
        color = 'green'
    else:
        raise ValueError(f'{vtype}')

    try:
        fg_color = getattr(Fore, color.upper())
    except AttributeError:
        fg_color = Fore.WHITE

    msg = fg_color + prefix
    if timestamp:
        msg += f"[{datetime.today().strftime('%y-%m-%d %H:%M:%S')}] - "

    try:
        out = f"{''.join(msgs)}\n"
    except TypeError:
        out = f'{msgs}'

    msg += out
    msg += Style.RESET_ALL
    print(msg, **kwarg)
