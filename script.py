import glob
import pickle
import matplotlib.pyplot as plt
import os

def load_and_show_figure(filename):
    # os.chdir('webmail_plots/')
    print(filename)
    with open(filename, 'rb') as f:
        fig = pickle.load(f)
        plt.show()

# LOading pickle object to check, remove later

os.chdir('webmail_plots/')
for file in glob.glob('./*'): 
    print(file)
    load_and_show_figure(file)