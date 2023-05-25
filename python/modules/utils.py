import math
import os
import sys

from scipy import stats as stats


def print_analysis_name(name):
    [print() for _ in range(3)]
    print('***-----***-----***-----***-----***-----***-----***-----***-----***')
    print(f' |{name : ^63}|')
    print('***-----***-----***-----***-----***-----***-----***-----***-----***')
    print()


def stdize(s):  # helps avoid convergence errors
    return (s - s.mean()) / s.std()


class HidePrints:
    # from: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def paired_t_test(s0, s1, get_p=False):
    N = len(s0)
    dif = s0 - s1
    d = dif.mean() / dif.std()

    t, p = stats.ttest_rel(s0, s1)

    print(f's0: M = {s0.mean():.4f}, SD = {s0.std() / math.sqrt(len(s0)):.4f}')
    print(f's1: M = {s1.mean():.4f}, SD = {s1.std() / math.sqrt(len(s1)):.4f}')
    print(f'\tDifference M = {dif.mean():.3f}')
    print(f't[{N - 1}] = {t:.3f}, {p=:.3f}')
    print(f'{colors.GREEN}\td = {d:.3f}{colors.ENDC}')
    wilcox = stats.wilcoxon(s0, s1)  # Although not reported in the paper, we also confirmed the significant t-test
    print(f'{wilcox.statistic=:.3f}, {wilcox.pvalue=:.3f}')  # results pan out per a non-parametric test

    if get_p:
        return p
    else:
        return d


def mean(l):
    return sum(l) / len(l)


class colors:
    '''
    Codes for printing colored text to the terminal. Helps things look good
    Taken from: https://stackoverflow.com/questions/37340049/how-do-i-print-colored-output-to-the-terminal-in-python
    '''
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
