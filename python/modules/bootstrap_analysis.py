import os
import pathlib

import pandas as pd
from tqdm import tqdm

from python.modules.utils import HidePrints


def printout_ds(ds, nsims):
    ds.sort()
    p025 = int(nsims * .025)
    p975 = int(nsims * .975)
    d_low = ds[p025]
    d_mid = ds[int(nsims * .5)]
    d_high = ds[p975]
    print(f'{int(nsims):<3f} | {d_mid=:.3f} [{d_low=:.3f}, {d_high=:.3f}]')


def run_bootstrap_sim(df, func, fn, nsims=10000, multilevel=False,
                      **kwargs):
    ds = []
    for _ in tqdm(range(nsims), desc=f'bootstrapping: {func=}'):
        df_boot = bootstrap_df(df, multilevel=multilevel)
        with HidePrints():  # HidePrints suppresses print statements
            d = func(df_boot, **kwargs)
        ds.append(d)
    printout_ds(ds, nsims)
    dir_out = fr'{pathlib.Path(__file__).parent.parent.parent}\bootstrap_results'
    fp_out = os.path.join(dir_out, fn)
    ds = [f'{d:.7f}\n' for d in ds]
    with open(fp_out, 'w') as file:
        file.writelines(ds)

    [print() for _ in range(3)]
    return ds


def bootstrap_df(df, multilevel=False):
    df_grpd = df.groupby('subject')
    rand_subj = pd.Series(df.subject.unique()).sample(frac=1, replace=True)
    l_df_new = []
    for i, subj in enumerate(rand_subj):
        df_subj = df_grpd.get_group(subj)
        df_subj['Subject'] = i
        if multilevel:
            l_df_new.append(df_subj.sample(frac=1, replace=True))
        else:
            l_df_new.append(df_subj)
    df_new = pd.concat(l_df_new)
    return df_new
