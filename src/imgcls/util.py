from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
from colorama import Fore, Style

__all__ = ['uglob',
           'printdf',
           'fprint']


def uglob(directory: Path,
          pattern: str,
          sort: bool = True,
          is_dir: bool = False) -> Path:
    """
    Unique glob the pattern in a directory

    :param directory: directory
    :param pattern: pattern string
    :param sort: if sort
    :param is_dir: only return if is a directory
    :return: unique path
    """
    if not isinstance(directory, Path):
        directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f'{directory} not exit')

    if not directory.is_dir():
        raise NotADirectoryError(f'{directory} is not a directory')

    f = list(directory.glob(pattern))

    if is_dir:
        f = [ff for ff in f if ff.is_dir()]

    if sort:
        f.sort()

    if len(f) == 0:
        raise FileNotFoundError(f'{directory} not have pattern: {pattern}')
    elif len(f) == 1:
        return f[0]
    else:
        raise RuntimeError(f'multiple files were found in {directory} in pattern {pattern} >>> {f}')


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
