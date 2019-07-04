import solver

import csv
import time
import random
import logging
import math
import os

from parse import parse

MIN_NB_OF_TRUCKS = 5
MAX_NB_OF_TRUCKS = 5

MIN_TRUCK_CAPACITY = 1000
MAX_TRUCK_CAPACITY = 1000

MIN_NB_OF_NODES = 100
MAX_NB_OF_NODES = 100

MIN_NODE_Y = 0
MAX_NODE_Y = 100

MIN_NODE_X = 0
MAX_NODE_X = 100

MIN_DEMAND = 1
MAX_DEMAND = 3

logging.basicConfig(filename='logs.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def raiseIfNone(var):
    if var is None:
        raise Exception('Cannot be None')


def generateFromSeed(seed=None, varName=None, varValue=None):
    if seed is None:
        random.seed()
        seed = random.random()

    random.seed(seed)
    logging.info('Seed : {0}'.format(seed))
    logging.info('Setup : {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}'.format(MIN_NB_OF_TRUCKS,
                                                                                               MAX_NB_OF_TRUCKS,
                                                                                               MIN_TRUCK_CAPACITY,
                                                                                               MAX_TRUCK_CAPACITY,
                                                                                               MIN_NB_OF_NODES,
                                                                                               MAX_NB_OF_NODES,
                                                                                               MIN_NODE_X,
                                                                                               MAX_NODE_X,
                                                                                               MIN_NODE_Y,
                                                                                               MAX_NODE_Y,
                                                                                               MIN_DEMAND,
                                                                                               MAX_DEMAND))

    citiesCount = random.randint(MIN_NB_OF_NODES, MAX_NB_OF_NODES)
    if varName == 'citiesCount':
        citiesCount = varValue

    instance = {
        'trucksCount': random.randint(MIN_NB_OF_TRUCKS, MAX_NB_OF_TRUCKS),
        'trucksCapacity': random.randint(MIN_TRUCK_CAPACITY, MAX_TRUCK_CAPACITY),
        'nodes': [],
        'matrix': [[0]*citiesCount for i in range(citiesCount)]
    }

    if varName == 'trucksCount':
        instance['trucksCount'] = varValue

    totalDemand = 0
    totalDeliveryCapacity = instance['trucksCount'] * instance['trucksCapacity']

    for i in range(0, citiesCount):
        nodeX = random.randint(MIN_NODE_X, MAX_NODE_X)
        nodeY = random.randint(MIN_NODE_Y, MAX_NODE_Y)
        demand = random.randint(MIN_DEMAND, MAX_DEMAND)
        instance['nodes'].append({'id': i, 'x': nodeX, 'y': nodeY, 'demand': demand})

        totalDemand += demand
        if totalDemand > totalDeliveryCapacity:
            instance['trucksCapacity'] += math.ceil((totalDemand - totalDeliveryCapacity) / instance['trucksCount'])

    for fromNode in instance['nodes']:
        for toNode in instance['nodes'][fromNode['id']:citiesCount]:
            dist = int(math.hypot(toNode['x'] - fromNode['x'], toNode['y'] - fromNode['y']))
            instance['matrix'][fromNode['id']][toNode['id']] = dist
            instance['matrix'][toNode['id']][fromNode['id']] = dist

    return instance


def retrieveFromFile(fileName):
    instance = {
        'nodes': list()
    }

    citiesCount = None

    with open(fileName, 'rt') as myFile:
        if myFile is None:
            raise Exception('Cannot open file \'{0}\''.format(fileName))

        logging.info('FileName : {0}'.format(fileName))

        line = myFile.readline()
        while line:
            if line.startswith("NAME : "):
                instance['name'] = parse('NAME : {}', line)[0]
            elif line.startswith("COMMENT : "):
                temp = parse('COMMENT : {} No of trucks: {}, Optimal value: {})', line)
                instance['trucksCount'] = int(temp[1])
                instance['optimalValue'] = int(temp[2])
            elif line.startswith("TYPE : "):
                pass
            elif line.startswith("DIMENSION : "):
                citiesCount = int(parse('DIMENSION : {}', line)[0])
            elif line.startswith("EDGE_WEIGHT_TYPE : "):
                pass
            elif line.startswith("CAPACITY : "):
                instance['trucksCapacity'] = int(parse('CAPACITY : {}', line)[0])
            elif line.startswith("NODE_COORD_SECTION "):
                raiseIfNone(citiesCount)
                for i in range(0, citiesCount):
                    line = myFile.readline()
                    id, x, y = parse(' {} {} {}', line)
                    instance['nodes'].append({'id': int(id) - 1, 'x': int(x), 'y': int(y)})

            elif line.startswith("DEMAND_SECTION "):
                raiseIfNone(citiesCount)
                for i in range(0, citiesCount):
                    line = myFile.readline()
                    id, demand = parse('{} {}', line)
                    instance['nodes'][int(id) - 1]['demand'] = int(demand[:-1])

            elif line.startswith("DEPOT_SECTION "):
                line = myFile.readline()
                instance['depotNodeId'] = int(parse(' {}', line)[0]) - 1
                break

            line = myFile.readline()

    if citiesCount is None:
        raise Exception('CitiesCount cannot be None, verify that the input file as a valid format')

    instance['matrix'] = [[0]*citiesCount for i in range(citiesCount)]

    for fromNode in instance['nodes']:
        for toNode in instance['nodes'][fromNode['id']:citiesCount]:
            dist = int(math.hypot(toNode['x'] - fromNode['x'], toNode['y'] - fromNode['y']))
            instance['matrix'][fromNode['id']][toNode['id']] = dist
            instance['matrix'][toNode['id']][fromNode['id']] = dist

    return instance


def generateStatsRandom(min, max, pas, varName, fileName):
    file = open(fileName, "w+", newline='')
    try:
        writer = csv.writer(file)
        writer.writerow(('Time', 'Result', 'MeanResult', 'ExpectedResult', 'CitiesCount', 'TrucksCount', 'Iterations', 'Temperature', 'Coef'))

        for i in range(min, max, pas):
            instance = generateFromSeed(varName=varName, varValue=i)

            expectedResult = None
            iterations = 10000
            temperature = 10
            coef = 0.95

            start_time = time.time()

            weights = {}
            for demand in instance['nodes']:
                weights[demand.get('id')] = int(demand.get('demand'))

            current_optimum, mean, myresults, allmyresults = solver.recuitCVRPGraph(instance['matrix'], weights, instance['trucksCount'], instance['trucksCapacity'], temperature, coef, iterations)

            writer.writerow((time.time() - start_time, current_optimum, mean, expectedResult, len(instance['nodes']), instance['trucksCount'], iterations, temperature, coef))
            print("Done {} {}".format(i, varName))
    finally:
        file.close()


def generateStatsQuality(values, valueName, inputFile, outputFile):
    file = open(outputFile, "w+", newline='')
    try:
        writer = csv.writer(file)
        writer.writerow(('Time', 'Result', 'MeanResult', 'ExpectedResult', 'CitiesCount', 'TrucksCount', 'Iterations', 'Temperature', 'Coef'))

        for i in values:
            instance = retrieveFromFile(inputFile)

            iterations = 1000000
            temperature = 10
            coef = 0.95

            if valueName == 'temperature':
                temperature = i
            elif valueName == 'iterations':
                iterations = i
            elif valueName == 'coef':
                temperature = 100
                coef = i

            start_time = time.time()

            weights = {}
            for demand in instance['nodes']:
                weights[demand.get('id')] = int(demand.get('demand'))

            current_optimum, mean, myresults, allmyresults = solver.recuitCVRPGraph(instance['matrix'], weights, instance['trucksCount'], instance['trucksCapacity'], temperature, coef, iterations)

            writer.writerow((time.time() - start_time, current_optimum, mean, instance['optimalValue'], len(instance['nodes']), instance['trucksCount'], iterations, temperature, coef))
            print("Done {0} {1}".format(valueName, i))
    finally:
        file.close()


def main():
    dirName = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    generateStatsRandom(10, 2000, 100, 'citiesCount', dirName + '/cities.csv')
    generateStatsRandom(6, 30, 1, 'trucksCount', dirName + '/trucks.csv')
    generateStatsQuality([1, 2, 5, 10], 'temperature', '../input0.txt', dirName + '/temperature.csv')
    generateStatsQuality([10000, 100000, 1000000], 'iterations', '../input0.txt', dirName + '/iterations.csv')
    generateStatsQuality([0.9, 0.95, 0.975, 0.99, 0.999], 'coef', '../input0.txt', dirName + '/coef.csv')


if __name__ == "__main__":
    main()
