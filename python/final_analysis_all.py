import os
import pathlib
import sys

# imports will work regardless of the directory used to call script
sys.path.append(f'{pathlib.Path(__file__).parent.parent}')

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import warnings
import numpy as np

from python.modules.bootstrap_analysis import run_bootstrap_sim
from python.modules.study1_tests import study1_EmoRating_on_gaze, study1_Emo_vs_Neu
from python.modules.study2_tests import study2_Emo_vs_Neu, study2_FG_vs_BG, study2_EmoRating_on_gaze

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
# For the record, you generally shouldn't suppress warnings willy nilly

np.random.seed(0)  # For reproducible bootstrapping. Notably, using random.seed()


# from the Python standard library will not work because
# boostrapping uses Pandas' DataFrame.sample(), which uses
# numpy for its randomization.

def run_study1(online=True, bootstrap=False, nsims=10_000):
    dir_in = r'in_data'
    dir_in = os.path.join(pathlib.Path(__file__).parent.parent, dir_in)
    if online:
        fn_in = 'Study1_online_data_anonymized_gaze.csv'
    else:
        raise ValueError(
            'We did not receive permission from our IRB to release the in-person data due to issues with the consent form.')
    fp_in = os.path.join(dir_in, fn_in)
    df = pd.read_csv(fp_in)
    df = df[~df['excluded'].astype(bool)]
    df['BG_propo'] = 1 - df['FG_propo']

    if bootstrap:
        run_bootstrap_sim(df, study1_Emo_vs_Neu,
                          f'Study1_Emo_vs_Neu_online_{nsims}.txt',
                          nsims=nsims, multilevel=False)
        run_bootstrap_sim(df, study1_EmoRating_on_gaze,
                          f'Study1_EmoRating_on_gaze_online_{nsims}.txt',
                          control_emo=False, nsims=nsims, multilevel=True)
        run_bootstrap_sim(df, study1_EmoRating_on_gaze,
                          f'Study1_EmoRating_on_gaze_ctrl_emo_online_{nsims}.txt',
                          control_emo=True, nsims=nsims, multilevel=True)
    else:
        form = 'online' if online else 'in-person'
        study1_Emo_vs_Neu(df, form, plot=False)
        study1_EmoRating_on_gaze(df, form, control_emo=False, plot=False,
                                 title=f'Study 1 ({form}): Impact on EmoRating',
                                 vmax=4)
        study1_EmoRating_on_gaze(df, form, control_emo=True, plot=False,
                                 title=f'Study 1 ({form}): Impact on EmoRating',
                                 vmax=4)


def run_study2(online=True, bootstrap=False, nsims=10_000):
    dir_in = r'in_data'
    dir_in = os.path.join(pathlib.Path(__file__).parent.parent, dir_in)
    if online:
        fn_in = 'Study2_online_data_anonymized_gaze.csv'
    else:
        raise ValueError(
            'We did not receive permission from our IRB to release the in-person data due to issues with the consent form.')
    fp_in = os.path.join(dir_in, fn_in)
    df = pd.read_csv(fp_in)
    df = df[~df['excluded'].astype(bool)]
    df['BG_propo'] = 1 - df['FG_propo']
    form = 'online' if online else 'in-person'

    if bootstrap:
        run_bootstrap_sim(df, study2_FG_vs_BG,
                          f'study2_FG_vs_BG_online_{nsims}.txt',
                          nsims=nsims, multilevel=False)
        run_bootstrap_sim(df, study2_Emo_vs_Neu,
                          f'study2_EmoBG_vs_NeuBG_online_{nsims}.txt',
                          FG_BG='BG', nsims=nsims, multilevel=False)
        run_bootstrap_sim(df, study2_Emo_vs_Neu,
                          f'study2_EmoFG_vs_NeuFG_online_{nsims}.txt',
                          FG_BG='FG', nsims=nsims, multilevel=False)
        run_bootstrap_sim(df, study2_EmoRating_on_gaze,
                          f'study2_EmoRating_on_gaze_emo_online_{nsims}.txt',
                          cond='emo', nsims=nsims, multilevel=True)
    else:
        study2_FG_vs_BG(df, form=form, plot=False,
                        title=f'Study 2 ({form}): FG vs. BG')
        study2_Emo_vs_Neu(df, FG_BG='BG', form=form, plot=False,
                          title=f'Study 2 ({form}): EmoBG vs. NeuBG')

        study2_Emo_vs_Neu(df, FG_BG='FG', form=form, plot=False,
                          title=f'Study 2 ({form}): EmoFG vs. NeuFG')
        study2_EmoRating_on_gaze(df, cond='emo', form=form, plot=False,
                                 title=f'Study 2 ({form}; emotion): Impact on EmoRating')
        study2_EmoRating_on_gaze(df, cond='neu', form=form, plot=False,
                                 title=f'Study 2 ({form}; neutral): Impact on EmoRating')


if __name__ == '__main__':
    run_study1(online=True, bootstrap=False, nsims=10_000)  # turn bootstrap=True to generate the dz distributions
    run_study2(online=True, bootstrap=False, nsims=10_000)
