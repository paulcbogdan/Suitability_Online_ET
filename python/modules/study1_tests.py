import numpy as np

from python.modules.plotting_results import plot_bar, plot_lmer
from python.modules.utils import print_analysis_name, paired_t_test, stdize, colors


def study1_Emo_vs_Neu(df, form='', plot=False):
    title = f'Study 1 ({form}): Emo vs. Neu'
    print_analysis_name(title)
    groupby_code = ['subject', 'emotionality']
    df = df.dropna(subset=['FG_propo'])
    df_study_avg = df.groupby(by=groupby_code).mean()
    s_Emo = df_study_avg.loc[(slice(None), 'emo'), 'FG_propo'].droplevel(1)
    s_Neu = df_study_avg.loc[(slice(None), 'neu'), 'FG_propo'].droplevel(1)

    d_or_p = paired_t_test(s_Emo, s_Neu, get_p=plot)
    if plot:
        plot_bar(s_Emo, s_Neu, title, d_or_p)

    return d_or_p


def study1_EmoRating_on_gaze(df, form='', control_emo=False, plot=False,
                             title='', vmax=4.):
    from pymer4.models import Lmer
    if control_emo:
        print_analysis_name(f'Study 1 {form}: EmoRating ~ FG_propo + Emotionality')
        formula = 'EmoRating ~ 1 + FG_propo + emotionality + ' \
                  '(1 + FG_propo + emotionality | subject)'
        variable_names = ['EmoRating', 'FG_propo', 'subject', 'emotionality']
    else:
        print_analysis_name(f'Study 1 {form}: EmoRating ~ FG_propo')
        formula = 'EmoRating ~ 1 + FG_propo + (1 + FG_propo | subject)'
        variable_names = ['EmoRating', 'FG_propo', 'subject']

    n_trials_pre = len(df)
    df = df.dropna(subset=['EmoRating']) # No response
    n_trials_post = len(df)
    print(f'Number of trials with no EmoRating: {n_trials_pre - n_trials_post}')
    if not plot:
        df['EmoRating'] = stdize(df['EmoRating'])
        df['FG_propo'] = stdize(df['FG_propo'])

    model = Lmer(formula, data=df[variable_names].dropna())
    summary = model.fit(REML=False) # REML=False is needed to properly evaluate
                                    # the significance of fixed effects (see
                                    # Meteyard & Davies, 2022)
    t_of_interest = summary['T-stat'].loc['FG_propo']
    print(summary)
    if plot:
        plot_lmer(summary, title, label='Both', vmax=vmax)

    d_of_interest = t_of_interest / np.sqrt(len(df.subject.unique()))
    print(f'{colors.GREEN}{d_of_interest=:.3f}{colors.ENDC}')
    return d_of_interest
