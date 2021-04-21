#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if (__name__ == '__main__'):
    output_dir = sys.argv[1]
    input_curve_file = sys.argv[2]
    input_hist_file = sys.argv[3]
    title = sys.argv[4]
    curve_df = pd.read_csv(input_curve_file, delimiter = ' ')
    hist_df = pd.read_csv(input_hist_file)
    fig, ax = plt.subplots(1, 1, figsize = [3.0, 1.2])
    n, bins, patches = plt.hist(hist_df['y'], bins = 100, range = [0, 1000], weights = np.ones(len(hist_df['y'])) / 10 / len(hist_df['y']))
    ax.plot(curve_df['x'], curve_df['y'])
    ax.set_title(title)
    # plt.plot([1, 2, 3], [3, 9, 4], 'k')
    plt.savefig(output_dir)