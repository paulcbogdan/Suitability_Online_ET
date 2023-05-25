import numpy as np

from python.modules.plotting_results import plot_bar, plot_lmer
from python.modules.utils import print_analysis_name, paired_t_test, stdize, colors


def study2_FG_vs_BG(df, form='', plot=None, title=''):
    print_analysis_name(f'Study 2 ({form}): FG vs. BG (t-test)')
    groupby_code = ['subject', 'attn']
    df = df.dropna(subset=['FG_propo'])
    df_study_avg = df.groupby(by=groupby_code).mean()
    s_FG = df_study_avg.loc[(slice(None), 'FG'), 'FG_propo'].droplevel(1)
    s_BG = df_study_avg.loc[(slice(None), 'BG'), 'FG_propo'].droplevel(1)
    d_or_p = paired_t_test(s_FG, s_BG, get_p=plot)
    sns = set(s_FG.index.get_level_values(0)).intersection(
        s_BG.index.get_level_values(0))

    s_FG = s_FG.loc[sns]
    s_BG = s_BG.loc[sns]

    if plot:
        plot_bar(s_FG, s_BG, title, d_or_p)

    return d_or_p


def study2_Emo_vs_Neu(df, FG_BG='FG', form='', plot=False, title=''):
    print_analysis_name(f'Study 2 ({form}): Emo{FG_BG} vs. Neu{FG_BG}')
    groupby_code = ['subject', 'emotionality', 'attn']
    df = df.dropna(subset=['FG_propo'])
    df_study_avg = df.groupby(by=groupby_code).mean()
    s_Emo = df_study_avg.loc[(slice(None), 'emo', FG_BG), 'FG_propo'].droplevel(1)
    s_Neu = df_study_avg.loc[(slice(None), 'neu', FG_BG), 'FG_propo'].droplevel(1)
    sns = set(s_Emo.index.get_level_values(0)).intersection(s_Neu.index.get_level_values(0))

    s_Emo = s_Emo.loc[sns]
    s_Neu = s_Neu.loc[sns]

    d_or_p = paired_t_test(s_Emo, s_Neu, get_p=plot)
    if plot:
        plot_bar(s_Emo, s_Neu, title, d_or_p)

    return d_or_p


def study2_EmoRating_on_gaze(df, cond, form='', plot=False,
                             title=''):
    from pymer4.models import Lmer
    assert cond in ['emo', 'neu'], f'Invalid cond: {cond=}'
    df = df[df['emotionality'] == cond]
    print_analysis_name(f'Study 2 ({form}): EmoRating ~ BG_propo + attn ({cond=})')
    formula = 'EmoRating ~ 1 + BG_propo + attn + (1 + BG_propo + attn | subject)'
    variable_names = ['EmoRating', 'BG_propo', 'attn', 'subject']

    n_trials_pre = len(df)
    df = df.dropna(subset=['EmoRating']) # No response
    n_trials_post = len(df)
    print(f'Number of trials with no EmoRating: {n_trials_pre - n_trials_post}')

    if not plot:
        df['EmoRating'] = stdize(df['EmoRating'])
        df['FG_propo'] = stdize(df['FG_propo'])
    model = Lmer(formula, data=df[variable_names].dropna())
    summary = model.fit(REML=False,) # REML=False is needed to properly evaluate
                                     # the significance of fixed effects (see
                                     # Meteyard & Davies, 2022)
    if plot:
        plot_lmer(summary, title,
                  label='Emotion' if cond == 'emo' else 'Neutral')

    print(summary)
    t_of_interest = summary['T-stat'].loc['BG_propo']
    d_of_interest = t_of_interest / np.sqrt(len(df.subject.unique()))
    print(f'{colors.GREEN}{d_of_interest=:.3f}{colors.ENDC}')
    return d_of_interest
