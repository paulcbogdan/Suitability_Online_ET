import pathlib
import sys

# imports will work regardless of the directory used to call script
sys.path.append(f'{pathlib.Path(__file__).parent.parent}')

import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib


def read_ds(fp):
    with open(fp, 'r') as file:
        lines = [float(line.rstrip()) for line in file]
    return lines


def proc_ds(d_inperson, M_online, ds_online, x, label, color):
    SD_online = np.std(ds_online)  # Conforms the bootstrapped distribution to a normal distribution for the aggregation
    M_P = abs(M_online / d_inperson)
    SD_P = abs(SD_online / d_inperson)
    print(f'd: {M_online:.2f}, [{M_online - 1.96 * SD_online:.2f}, {M_online + 1.96 * SD_online:.2f}]')
    print(f'percent: {M_P:.1%} [{M_P - 1.96 * SD_P:.1%}, {M_P + 1.96 * SD_P:.1%}]')
    pdf = stats.norm.pdf(x, loc=abs(M_P),
                         scale=abs(SD_P))
    pdf = pdf / np.sum(pdf) * len(x)
    plt.plot(x * 100, pdf, label=label, color=color)
    return pdf


if __name__ == '__main__':
    matplotlib.rc('font', family='Arial', size=15)

    dir_in = r'bootstrap_results'
    dir_in = os.path.join(pathlib.Path(__file__).parent.parent, dir_in)

    nsims = 1000
    fns = [
        'Study1_Emo_vs_Neu_{}_{}.txt',
        'Study1_EmoRating_on_gaze_{}_{}.txt',
        'Study1_EmoRating_on_gaze_ctrl_emo_{}_{}.txt',
        'study2_FG_vs_BG_{}_{}.txt',
        'study2_EmoBG_vs_NeuBG_{}_{}.txt',
        'study2_EmoFG_vs_NeuFG_{}_{}.txt',
        'study2_EmoRating_on_gaze_emo_{}_{}.txt'
    ]

    labels = ['1. Emotion vs. Neutral',
              '2. EmoRating',
              '3. EmoRating (Control Emo)',
              '4. Attentional Cue',
              '5. Emotion Capture (BG)',
              '6. Emotion Capture (FG)',
              '7. EmoRating (Emo Only)',
              ]

    colors = ['royalblue',
              'orange',
              'green',
              'c',
              'red',
              'purple',
              'saddlebrown']

    d_inpersons = [1.140,
                   0.987,
                   0.453,
                   6.621,
                   0.706,
                   np.nan,
                   0.411
                   ]

    d_onlines = [0.614,
                 0.524,
                 0.232,
                 0.918,
                 0.349,
                 np.nan,
                 0.220]

    pdfs = []
    x = np.linspace(0., 1., 1001)

    efs = [0, 1, 2, 4, 6]  # final
    for ef in efs:
        fn = fns[ef]
        label = labels[ef]
        color = colors[ef]
        fp_online = os.path.join(dir_in, fn.format('online', 10_000))
        ds_online = read_ds(fp_online)
        d_inperson = d_inpersons[ef]
        d_online = d_onlines[ef]
        pdf = proc_ds(d_inperson, d_online, ds_online, x, label, color)
        pdfs.append(pdf)
    pdf_all = np.product(np.array(pdfs), axis=0)
    sum_pdf_all = sum(pdf_all)
    pdf_all = pdf_all / sum_pdf_all * len(x)

    cdf_all = np.cumsum(pdf_all / np.sum(pdf_all))
    lo, mid, hi = .025, .5, .975
    lo_i, mid_i, hi_i = None, None, None
    for i in range(len(cdf_all)):
        if lo_i is None and cdf_all[i] > lo:
            lo_i = i / len(cdf_all)
        if hi_i is None and cdf_all[i] > hi:
            hi_i = i / len(cdf_all)
        if mid_i is None and cdf_all[i] > mid:
            mid_i = i / len(cdf_all)

    max_plot = np.max(pdf_all)
    if len(efs) == 5:
        max_plot *= 3 / 5

    [print() for _ in range(3)]
    results = f'.025 = {lo_i:.3f} | {mid_i:.3f} | .975 = {hi_i:.3f}'

    plt.xlim(0, 100)
    plt.plot(x * 100, pdf_all, label='Aggregate', color='k', linewidth=3)
    plt.plot([lo_i * 100, lo_i * 100], [0, max_plot], color='k', linestyle='--')
    plt.plot([hi_i * 100, hi_i * 100], [0, max_plot], color='k', linestyle='--')
    plt.ylabel('Probability Density')
    plt.xlabel('Online $d_{z}$ / In-Person $d_{z}$ (%)')

    plt.legend(frameon=False, fontsize=11.5, borderpad=0.02)
    plt.gca().set_ylim(bottom=0)
    plt.tight_layout()

    dir_fig = f'{pathlib.Path(__file__).parent.parent}'
    ef_code = "".join(str(ef) for ef in efs)
    fp_fig = os.path.join(dir_fig, f'aggregated_effect_sizes_{ef_code}.png')

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(fp_fig, dpi=300)
    plt.show(dpi=300)
