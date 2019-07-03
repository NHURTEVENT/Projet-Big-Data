import solver

import csv
import time
import random
import logging
import math

RESULTS_FILE = time.strftime("%Y%m%d-%H%M%S") + '.csv'

MIN_NB_OF_TRUCKS = 2
MAX_NB_OF_TRUCKS = 5

MIN_TRUCK_CAPACITY = 50
MAX_TRUCK_CAPACITY = 100

MIN_NB_OF_NODES = 10
MAX_NB_OF_NODES = 10

MIN_NODE_Y = 0
MAX_NODE_Y = 100

MIN_NODE_X = 0
MAX_NODE_X = 100

MIN_DEMAND = 1
MAX_DEMAND = 25

logging.basicConfig(filename='logs.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def generateFromSeed(seed=None, citiesCountParameter=None):
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
    if citiesCountParameter is not None:
        citiesCount = citiesCountParameter

    instance = {
        'trucksCount': random.randint(MIN_NB_OF_TRUCKS, MAX_NB_OF_TRUCKS),
        'trucksCapacity': random.randint(MIN_TRUCK_CAPACITY, MAX_TRUCK_CAPACITY),
        'nodes': [],
        'matrix': [[0]*citiesCount for i in range(citiesCount)]
    }

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


def main():
    file = open(RESULTS_FILE, "w+", newline='')
    try:
        writer = csv.writer(file)
        writer.writerow(('Time', 'Result', 'ExpectedResult', 'CitiesCount', 'TrucksCount', 'Iterations', 'Temperature', 'Coef'))
        
        for i in range(10, 50):
            # put this block in a for loop
            instance = generateFromSeed(citiesCountParameter=i)

            expectedResult = 500
            iterations = 1000
            temperature = 10
            coef = 0.99

            start_time = time.time()
            
            weights = {}
            for demand in instance['nodes']:
                weights[demand.get('id')] = int(demand.get('demand'))

            current_optimum, mean, myresults, allmyresults = solver.recuitCVRPGraph(instance['matrix'], weights, instance['trucksCount'], instance['trucksCapacity'], temperature, coef, iterations)

            writer.writerow((time.time() - start_time, current_optimum, expectedResult, len(instance['nodes']), instance['trucksCount'], iterations, temperature, coef))
    finally:
        file.close()


if __name__ == "__main__":
    main()