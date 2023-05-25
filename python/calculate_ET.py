import os
import pathlib
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(f'{pathlib.Path(__file__).parent.parent}')

from python.modules.utils import mean


def load_IAs():
    # this .pkl file contains all the information from the .ias (eyelinke interest area files)
    fp_IAs = rf'{pathlib.Path(__file__).parent.parent}\interest_areas\FG_rectangle_IA_dict.pkl'
    with open(fp_IAs, 'rb') as file:
        FG_rectangle_IA_dict = pickle.load(file)
    IA_ars_dict = FG_rectangle_IA_dict['freehand']  # foreground
    rectangle_IA_dict = FG_rectangle_IA_dict['rectangle']  # background (rectangle)
    return IA_ars_dict, rectangle_IA_dict


def load_data(study):
    dir_in = f'{pathlib.Path(__file__).parent.parent}\in_data\eye_timeseries\study{study}'
    fns = os.listdir(dir_in)
    sn2data = {}
    for fn in fns:
        fp_subj = os.path.join(dir_in, fn)
        with open(fp_subj, 'rb') as file:
            sn2data[int(fn.replace('.pkl', ''))] = pickle.load(file)
    return sn2data


def is_point_in_IA(point, IA_ars, rectangle=None):
    def isPointInPath(x, y, poly):
        # copied from: https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule
        """
        x, y -- x and y coordinates of point
        poly -- a list of tuples [(x, y), (x, y), ...]
        """
        num = len(poly)
        i = 0
        j = num - 1
        c = False
        for i in range(num):
            if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                    (x < poly[i][0] + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
                     (poly[j][1] - poly[i][1])):
                c = not c
            j = i
        return c

    if rectangle is None:
        # thus IA_ars = rectangle_IA_ar
        x = point['x']
        y = point['y']
        is_in = False
        if isPointInPath(x, y, IA_ars):
            is_in = True
        return is_in
    else:
        x = point['x']
        y = point['y']
        is_in = False
        if len(IA_ars[0]) > 2:
            for IA_ar in IA_ars:
                if isPointInPath(x, y, IA_ar):
                    is_in = True
        else:
            if isPointInPath(x, y, IA_ars):
                is_in = True
        return is_in


def get_point_in_ia_stats(all_points, ia_freehand, ia_rectangle, rect_border=.05):
    in_FGs = []
    in_BGs = []
    in_FG_points = []
    out_FG_points = []
    for idx, (x, y) in enumerate(all_points):
        if pd.isna(x) or pd.isna(y): continue
        in_FG = is_point_in_IA({'x': x, 'y': y}, ia_freehand, rectangle=ia_rectangle)
        x_mod = x - rect_border if x > .5 else x + rect_border
        y_mod = y - rect_border if y > .5 else y + rect_border
        in_BG = is_point_in_IA({'x': x_mod, 'y': y_mod}, ia_rectangle, rectangle=ia_rectangle)
        in_FGs.append(in_FG)
        in_BGs.append(in_BG)
        in_FG_points.append((x, y, idx)) if in_FG else out_FG_points.append((x, y, idx))
    return in_FGs, in_BGs, in_FG_points, out_FG_points


def add_gaze_scores(df, study=1):
    print('recalculate_ET...')
    ia_FGs_dict, ia_rectangle_dict = load_IAs()  # ia_FGs_dict[ia_name] = list of polygons representing the foreground
    # ia_rectangle_dict[ia_name] = list of polygons representing the image
    sn2pic2ET = load_data(study)  # dict of E-T data, where sn2pic2d[sn][pic] = list of coordinates for a given trial
    sn2pic2scores = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: np.nan)))

    # Calculate gaze scores (FG_time, BG_time, FG_propo) based on the E-T coordinate timeseries
    for sn, pic2d in tqdm(sn2pic2ET.items()):
        for pic, xy in pic2d.items():
            ia_FG = ia_FGs_dict[pic]
            ia_rectangle = ia_rectangle_dict[pic]
            rect_border = .05  # padding around rectangle for points that barely fall outside
            in_FGs, in_BGs, in_FG_points, out_FG_points = get_point_in_ia_stats(
                xy, ia_FG, ia_rectangle, rect_border=rect_border)
            if len(in_FGs) < 1:  # Not a single point recorded anywhere
                continue
            FG_time = mean(in_FGs)
            rect_time = mean(in_BGs)
            BG_time = rect_time - FG_time
            if FG_time + BG_time > 0.0001:  # Ensures at least one point recorded in either the FG or BG
                FG_propo = FG_time / (FG_time + BG_time)
            else:
                FG_propo = np.nan
            sn2pic2scores[sn][pic] = {'FG_time': FG_time,
                                      'BG_time': BG_time,
                                      'FG_propo': FG_propo}

    for key in ['FG_time', 'BG_time', 'FG_propo']:
        df[key] = df[['subject', 'IA_name']].apply(
            lambda row: sn2pic2scores[row['subject']][row['IA_name']][key],
            axis=1)

    return df


if __name__ == '__main__':
    fp_study1 = fr'{pathlib.Path(__file__).parent.parent}/in_data/Study1_online_data_anonymized.csv'
    df1 = pd.read_csv(fp_study1)
    df1_added = add_gaze_scores(df1, study=1)
    df1_added.to_csv(fr'{pathlib.Path(__file__).parent.parent}/in_data/Study1_online_data_anonymized_gaze.csv',
                     index=False)

    fp_study2 = fr'{pathlib.Path(__file__).parent.parent}/in_data/Study2_online_data_anonymized.csv'
    df2 = pd.read_csv(fp_study2)
    df2_added = add_gaze_scores(df2, study=2)
    df2_added.to_csv(fr'{pathlib.Path(__file__).parent.parent}/in_data/Study2_online_data_anonymized_gaze.csv',
                     index=False)
