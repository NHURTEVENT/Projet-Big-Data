import sys

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats

def calculateLinearRegression(xvalues, yvalues, xaxistitle, yaxistitle, windowTitle):
    xvalues = np.array(xvalues)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xvalues, yvalues)
    line = slope * xvalues + intercept

    print('Y(x) = {0}x + {1}'.format(slope, intercept))

    plt.plot(xvalues, yvalues, 'o', xvalues, line)
    plt.xlabel(xaxistitle)
    plt.ylabel(yaxistitle)
    #pylab.title(windowTitle)

    return slope, intercept


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
    RESULTS_DIRECTORY = 'Results/'
    
    data = retrieveStatsFromFile(RESULTS_DIRECTORY + 'cities.csv')
    plt.subplot(321)
    calculateLinearRegression([int(x[4]) for x in data], [float(x[0]) for x in data], 'Nb of cities', 'time (s)', 'Nb of cities / Time (s)')

    data = retrieveStatsFromFile(RESULTS_DIRECTORY + 'trucks.csv')
    plt.subplot(322)
    calculateLinearRegression([int(x[5]) for x in data], [float(x[0]) for x in data], 'Nb of trucks', 'time (s)', 'Nb of trucks / Time (s)')

    data = retrieveStatsFromFile(RESULTS_DIRECTORY + 'temperature.csv')
    plt.subplot(323)
    margeOptimum = [float(x[2]) - int(x[3]) for x in data]
    margeResult = [float(x[2]) - int(x[1]) for x in data]
    qualityCoef = [margeResult[i] / margeOptimum[i] for i in range(0, len(data))]
    coefSlope, coefIntercept = calculateLinearRegression([int(x[7]) for x in data], qualityCoef, 'Temperature', 'Quality Coef', 'Temperature / Quality Coef')

    data = retrieveStatsFromFile(RESULTS_DIRECTORY + 'coef.csv')
    plt.subplot(324)
    margeOptimum = [float(x[2]) - int(x[3]) for x in data]
    margeResult = [float(x[2]) - int(x[1]) for x in data]
    qualityCoef = [margeResult[i] / margeOptimum[i] for i in range(0, len(data))]
    temperatureSlope, temperatureIntercept = calculateLinearRegression([float(x[8]) for x in data], qualityCoef, 'Temperature Coef', 'Quality Coef', 'Coef / Quality Coef')

    data = retrieveStatsFromFile(RESULTS_DIRECTORY + 'iterations.csv')
    plt.subplot(325)
    margeOptimum = [float(x[2]) - int(x[3]) for x in data]
    margeResult = [float(x[2]) - int(x[1]) for x in data]
    qualityCoef = [margeResult[i] / margeOptimum[i] for i in range(0, len(data))]
    calculateLinearRegression([int(x[6]) for x in data], qualityCoef, 'Iterations', 'Quality Coef', 'Iterations / Quality Coef')

    plt.subplot(326)
    x = np.linspace(-5, 5, 100)
    y = (temperatureSlope + coefSlope) / 2 * x + (temperatureIntercept + coefSlope) / 2
    plt.plot(x, y, '-r') 

    plt.subplots_adjust(bottom=8, top=10, left=3, right=5)
    plt.show()

if __name__ == "__main__":
    main()