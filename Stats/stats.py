import sys

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats

def calculateLinearRegression(xvalues, yvalues, windowTitle):
    xvalues = np.array(xvalues)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xvalues, yvalues)
    line = slope * xvalues + intercept

    print('Y(x) = {0}x + {1}'.format(slope, intercept))

    plt.plot(xvalues, yvalues, 'o', xvalues, line)
    pylab.title(windowTitle)


def retrieveStatsFromFile(fileName):
    data = [[]]

    file = open(fileName, "rt")
    try:
        reader = csv.reader(file)
        next(reader)
        data = list(reader)
    finally:
        file.close()
    return data

def main():
    fileName = "20190704-002519.csv"
    data = retrieveStatsFromFile(fileName)
    
    calculateLinearRegression([int(x[3]) for x in data],[float(x[0]) for x in data], '')

if __name__ == "__main__":
    main()