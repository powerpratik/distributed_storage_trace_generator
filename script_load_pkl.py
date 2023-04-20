

import glob
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

from scipy import stats
import numpy as np

def load_and_show_figure(filename):
    # os.chdir('webmail_plots/')
    print(filename)
    with open(filename, 'rb') as f:
        fig = pickle.load(f)
        plt.show()

# LOading pickle object to check, remove later
print(matplotlib.__version__)
# load_and_show_figure('rt_best_fit_cdf.pkl')
os.chdir('nexus_plots/')
for file in glob.glob('./*'): 
    if 'cdf' in file:
        print(file)
        load_and_show_figure(file)


# rng = np.random.default_rng()
# print(stats.norm.rvs(size=100, random_state=rng))