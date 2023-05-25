import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def p_to_stars(p):
    fontsize = 20
    if p < .001:
        stars = '***'
    elif p < .01:
        stars = '**'
    elif p < .05:
        stars = '*'
    else:
        stars = f'NS (p = {p:.2f})'
        fontsize = 20
    return stars, fontsize


def plot_bar(s_Emo, s_Neu, title, p):
    matplotlib.rc('font', **{'size': 18})
    M_emo = s_Emo.mean()
    M_neu = s_Neu.mean()
    SE_emo = s_Emo.std() / math.sqrt(len(s_Emo))
    SE_neu = s_Neu.std() / math.sqrt(len(s_Emo))
    cols = ('Emotion', 'Neutral')
    plt.title(title, fontsize=18)
    plt.ylabel('FG proportion')
    plt.bar(cols, [M_emo, M_neu], width=0.995,
            label=cols, color=['red', 'royalblue'],
            yerr=[SE_emo, SE_neu])

    dh = .04
    barh = .053
    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)

    lx, ly = 0, M_emo + SE_emo
    rx, ry = 1, M_neu + SE_neu
    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]

    stars, fontsize = p_to_stars(p)
    if 'NS' in stars: barh -= .12
    mid = ((lx + rx) / 2, y - barh)

    plt.text(*mid, stars, fontsize=fontsize, ha='center', va='bottom')
    plt.plot(barx, bary, c='black')
    plt.ylim(0., 1.05)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.tight_layout()

    plt.show()


def plot_lmer(summary, title, label='',
              vmin=1, vmax=4.):
    matplotlib.rc('font', **{'size': 18})
    plt.title(title, fontsize=16 if len(title) < 40 else 14)

    intr = summary['Estimate'].loc['(Intercept)']
    gaze_key = 'FG_propo' if 'FG_propo' in summary['Estimate'] else 'BG_propo'
    slope = summary['Estimate'].loc[gaze_key]
    x = np.linspace(0, 1, 11)
    y_base = intr + x * slope
    label_to_color = {'Emotion': 'red', 'Neutral': 'royalblue',
                      '': 'k', 'Both': 'purple',
                      'Foreground focus': 'orange',
                      'Background focus': 'green'}

    p = summary['P-val'].loc[gaze_key]
    stars, fontsize = p_to_stars(p)

    if 'emotionalityneu' in summary['Estimate']:
        neu_cond = summary['Estimate'].loc['emotionalityneu']
        y_neu = intr + x * slope + neu_cond
        stars_y = intr + 0.5 * slope + neu_cond * 0.5
        plt.plot(x, y_base, label='Emotion', linewidth=3,
                 color=label_to_color['Emotion'])
        plt.plot(x, y_neu, label='Neutral', linewidth=3,
                 color=label_to_color['Neutral'])
    elif 'attnFG' in summary['Estimate']:
        neu_cond = summary['Estimate'].loc['attnFG']
        y_FG = intr + x * slope + neu_cond
        stars_y = intr + 0.5 * slope + neu_cond * 0.5
        plt.plot(x, y_FG, label='Foreground focus', linewidth=3,
                 color=label_to_color['Foreground focus'])
        plt.plot(x, y_base, label='Background focus', linewidth=3,
                 color=label_to_color['Background focus'])
    else:
        stars_y = intr + 0.5 * slope + 0.25
        plt.plot(x, y_base, linewidth=3, color=label_to_color[label])
    plt.text(0.5, stars_y, stars, fontsize=fontsize, ha='center',
             va='center')
    plt.ylabel('Emotional Rating')
    plt.ylim(vmin, vmax)
    plt.xlabel('Gaze within foreground')
    plt.legend(frameon=False)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.tight_layout()
    plt.show()
