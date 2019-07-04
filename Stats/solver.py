import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import networkx as nx
from pyvis.network import Network


def swap1(paths):
    # paths = paths.tolist()
    path1index = random.randrange(len(paths))
    path2index = random.randrange(len(paths))

    while len(paths[path2index]) == 0:
        path2index = random.randrange(len(paths))

    while path1index == path2index:
        path1index = random.randrange(len(paths))

    randP1 = math.ceil(random.random() * 2)
    randP2 = math.ceil(random.random() * 2)

    path1 = np.array(paths[path1index]).tolist()
    path2 = np.array(paths[path2index]).tolist()

    y1 = path2[0] if randP2 == 1 else path2[len(path2) - 1]
    path1.insert(0, y1)
    path2.remove(y1)
    paths[path1index] = path1
    paths[path2index] = path2
    return paths


def printError(x):
    print("random out of bound")

def f(x, matrice):
    total_sum = 0
    for cycle in x :
        #print("cycle : "+ str(cycle))
        cycle_sum = 0
        if(len(cycle)>0) :
            #print(range(0,len(cycle)))
            for i in range(0,len(cycle)) :
                if i == 0 :
                    cycle_sum += matrice[cycle[i]][0]
                    #print("0 to node "+ str(cycle[i])+" : "+ str(matrice[cycle[i]][0]))
                else :
                    cycle_sum += matrice[cycle[i]][cycle[i-1]]
                    #print("node "+str(cycle[i-1])+" to node "+str(cycle[i])+" : "+str(matrice[cycle[i]][cycle[i-1]]))

            #print("node "+ str(cycle[len(cycle)-1])+" to 0 : "+ str(matrice[cycle[len(cycle)-1]][0]))
            cycle_sum += matrice[cycle[len(cycle)-1]][0]
            #print("cycle sum "+ str(cycle_sum))
            total_sum += cycle_sum
    #print("total sum "+str(total_sum))
    return total_sum


def matriceToGraph(matrice):
    myGraph4 = nx.Graph()

    for i in range(len(matrice)):
        myGraph4.add_node(i)

    for i in range(len(matrice)) :
        for j in range(len(matrice)) :
            if(i>j) :
                #print(matrice[i][j])
                myGraph4.add_edge(i,j,weight=matrice[i][j])

    return myGraph4



def convertPaths(paths):
    newPaths = []
    for path in paths:
        newPath = []
        for cycle in path:
            cycle2 = []
            for node in cycle :
                cycle2.append(str(node))
            newPath.append(cycle2)
        #print("new path"+ str(newPath))
        newPaths.append(newPath)
    return newPaths

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#import seaborn.apionly as sns
import matplotlib.animation

fig, ax = plt.subplots(figsize=(7, 4))
G = None
pos = []
paths = []

def update(num):
    print("fig " + str(fig))
    print("ax" + str(ax))
    ax.clear()
    i = num // 3
    j = num % 3 + 1

    # cycles = paths[donePathNumber]

    newPaths = convertPaths(paths)
    cycles = newPaths[num]
    print(cycles)
    colors = ["red", "blue", "green", "black", "orange", "purple", "yellow", "brown"]
    index = 0
    for path in cycles:
        # Query nodes
        # set_edgecolor(colors[index%len(colors)])
        # set_edgecolor("red")

        index += 1
        print("path " + str(path))
        query_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=path, ax=ax)
        query_nodes.set_edgecolor("blue")
        nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path, path)), font_color="white", ax=ax)
        edgelist = [path[k:k + 2] for k in range(len(path) - 1)]
        print("edge list" + str(edgelist))
        edges = nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=3, ax=ax)
        edges.set_edgecolor(colors[index % len(colors)])
        # edges.set_edgecolor("red")

    # Scale plot ax
    ax.set_title("Frame %d:    " % (num + 1) + str(cycles), fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


# test(1)


# capaciteMax = 100
# allItems = {0:0,1:5,2:18,3:6,4:15,5:20,6:8,7:15,8:20,9:13}#, 10:21,11:25}
# print(allItems)

def getAllSwappableItems(currentPaths, capaciteMax, allItems):
    # print("current paths:" + str(currentPaths))
    # if not all full, verif plus peit poid swap possible
    full = True
    allMovesAvailable = []
    truckId = 1
    for cycle in currentPaths:
        truckWeight = 0
        # print("cycles"+ str(cycles))
        available = []
        # for cycle in cycles:

        # print("truck "+ str(truckId)+" " + str(cycle))
        for node in cycle:
            if (node == 0):
                continue

            truckWeight += allItems[node]
            # print("truckweight : " + str(truckWeight))

        # print("final truckweight : " + str(truckWeight))
        # print("max capa: "+ str(capaciteMax))
        if truckWeight < capaciteMax:
            freeSpace = capaciteMax - truckWeight
            # print("freespace "+str(freeSpace))
            full = False

            for index in allItems:
                if index not in cycle:
                    itemWeight = allItems.get(index)
                    # print("index"+ str(index))
                    # print("poid camion "+ str(truckWeight))
                    # print("poid item "+ str(allItems.get(index)))
                    if ((truckWeight + itemWeight) <= capaciteMax) and index != 0:
                        # same thing as
                        # if(itemWeight <= freeSpace)
                        # print("can add "+ str(index) +" of weight "+ str(itemWeight) + " to truck "+str(truckId)+" that has "+ str(freeSpace)+ " available sapce")
                        available.append(index)
                    # else :
                    # print("CAN'T add "+ str(index) +" of weight "+ str(itemWeight) + " to truck "+ str(truckId) + "that has "+ str(freeSpace)+ " available space")

                # [x for x in allItems if x not in cycle]
                # print(x)
        # print("all movables for cycle "+ str(cycle)+" : "+str(available))
        allMovesAvailable.append(available)
        truckId += 1
    # print("all movables this tick "+ str(allMovesAvailable))
    return allMovesAvailable


def allAreFull(paths, allItems, capaciteMax):
    full = True
    for path in paths:
        weight = 0
        for node in path:
            weight += allItems[node]
        if weight < capaciteMax:
            full = False
    return full


def swapExtraCVRP(paths, capaciteMax, allItems):  # , allItems, capaciteMax):

    swappableItem = getAllSwappableItems(paths, capaciteMax, allItems)

    # paths = paths.tolist()
    path1index = random.randrange(len(paths))
    indexTruckToLoad = random.randrange(len(paths))

    # path2index = random.randrange(len(paths))

    # while len(paths[path2index]) == 0 :
    #    path2index = random.randrange(len(paths))

    '''
    #ca devrait aussi marcher si on fait
    while len(swappableItem[path1index]) == 0:
        rerand1
    while path1index==path2index :
        rerand2
    '''

    for i in range(len(allItems) * 3):
        if len(swappableItem[indexTruckToLoad]) == 0:  # or len(paths[path2index]) == 0 or path1index==path2index:
            # path2index = random.randrange(len(paths))
            # print("in while")
            indexTruckToLoad = random.randrange(len(paths))
        else:
            # print("i'm in")
            break
        if i == len(allItems) * 3 - 1:
            # print("i'm out")
            return paths

    # print("swappable in truck "+str(indexTruckToLoad)+" : "+str(swappableItem[indexTruckToLoad]))

    truckToLoad = np.array(paths[indexTruckToLoad]).tolist()
    # path2 = np.array(paths[path2index]).tolist()

    insertIndex = random.randrange(0, len(truckToLoad) - 1) if len(truckToLoad) > 1 else 0
    extractIndex = random.randrange(0, len(swappableItem[indexTruckToLoad]) - 1) if len(
        swappableItem[indexTruckToLoad]) > 1 else 0

    # print("extract index "+str(extractIndex))

    # print("truck to load pre swap "+ str(truckToLoad)+" (truck "+str(indexTruckToLoad+10)+" )")
    # print("p2 pre swap "+ str(path2)+" (truck "+str(path2index+10)+" )")

    truckToUnload = []

    # take from the list of item possible to load in TruckToLoad
    # print("take from "+  str(swappableItem[indexTruckToLoad]))
    itemToLoad = swappableItem[indexTruckToLoad][extractIndex]

    # find path to extract from
    for truck in paths:
        if itemToLoad in truck:
            truckToUnload = np.array(truck).tolist()
            # print("truck to unload pre swap "+ str(truckToUnload))

    # print("truck to unload pre swap "+ str(truckToUnload))
    indexTruckToUnLoad = paths.index(truckToUnload)

    # = path2.tolist()
    # print("swap "+str(itemToLoad))
    truckToLoad.insert(insertIndex, itemToLoad)
    truckToUnload.remove(itemToLoad)
    paths[indexTruckToLoad] = truckToLoad
    paths[indexTruckToUnLoad] = truckToUnload

    # print("to load post swap "+ str(truckToLoad))
    # print("to unload post swap "+ str(truckToUnload))
    # print("")

    return paths


def swap2(paths, allItems):
    print("swap entrees intra")
    # paths = paths.tolist()
    path1index = random.randrange(len(paths))

    path1 = paths[path1index]

    while len(path1) < 2:
        path1index = random.randrange(len(paths))
        path1 = paths[path1index]
        print("in while 1")

    rand1 = random.randrange(len(path1))

    rand2 = random.randrange(len(path1))

    print("randrange " + str(len(path1)))
    print("rand1 " + str(rand1))
    print("rand2 " + str(rand2))
    print(path1)

    while rand1 == rand2:
        rand2 = random.randrange(len(path1))
        print("in while 2")
        print("regenerated rand2 " + str(rand2))

    x1 = path1[rand1]
    x2 = path1[rand2]
    # same si 1 element

    path1[rand1], path1[rand2] = path1[rand2], path1[rand1]

    paths[path1index] = path1

    # print(path1)
    # print(path2)
    # print("swaped extra")
    return paths

def switchAction(x):
    # print("x "+ str(x))
    switcher = {
        1: swapExtraCVRP,
        2: swap2

    }
    # print("xx "+ str(x))
    return switcher.get(x, printError)



def recuitCVRPGraph(matrice, allItems, k, capaciteMax, t, coef, n):
    paths = []
    current_optimum = 1000000000
    # xstart = np.array_split(range(1,len(matrice),1),k)
    xstart = splitClients(allItems, k, capaciteMax)
    xc = np.array(xstart)
    #print("xstart" + str(xstart))
    # print(xc)
    xc = xc.tolist()
    # print(xc)

    na = 0
    results = []
    allresults = []
    DeltaE_avg = 0.00000000001
    for i in range(n):
        randAction = random.randrange(1, 2)
        swap = switchAction(randAction)
        # swap = switchAction(2)
        # print("pre swap "+ str(xc))
        xc = swap(xc, capaciteMax, allItems)
        # print("post swap "+ str(xc))

        # if len(xc) > k:
        # print("It's impossible to deliver all items with "+str(k)+ " trucks, we need "+str(len(xc)) + "at minimum")
        if len(xc) < k:
            for j in range(0, k - len(xc)):
                xc.append([])
        # print("new xstart"+ str(xc))

        xi = xc
        totalCost = f(xi, matrice)
        # print("cost" + str(totalCost))

        DeltaE = abs(totalCost - current_optimum)
        if (totalCost > current_optimum):
            if (n == 0): DeltaE_avg = DeltaE
            # objective function is worse
            # generate probability of acceptance
            p = math.exp(-DeltaE / (DeltaE_avg * t))
            # determine whether to accept worse point
            if (random.random() < p):
                # accept the worse solution
                accept = True
            else:
                # don't accept the worse solution
                # print("nope")
                accept = False
        else:
            # objective function is lower, automatically accept
            accept = True
        if (accept == True):
            # update currently accepted solution
            xc = xi
            current_optimum = totalCost
            optimal_path = xc.copy()
            xopti2 = []  # xopti.copy()
            # print("accepted "+ str(xc))

            for path in optimal_path:
                path2 = np.array(path.copy()).tolist()
                # print("path2" + str(path2))
                path2.insert(0, 0)
                path2.insert(len(path2), 0)
                xopti2.append(path2)

            paths.append(xopti2)
            # paths.append(xopti2)
            # print("fc = "+ str(fc))
            results.append(current_optimum)
            # increment number of accepted solutions
            na = na + 1.0
            # update DeltaE_avg
            DeltaE_avg = (DeltaE_avg * (na - 1.0) + DeltaE) / na
        allresults.append(totalCost)
        t = coef * t
        # print("iter")

    #print(current_optimum)
    #print(optimal_path)
    mean_cost = sum(allresults)/ len(allresults)

     #return paths, results, allresults
    return current_optimum, mean_cost, results, allresults






def splitClients(allItems, k, capaciteMax):
    splitList = []
    currentWeight = 0
    totalWeight = 0

    # print("allItems" + str(allItems))
    #print(allItems)
    for itemIndex in allItems:
        totalWeight += allItems[itemIndex]

    averageWeight = totalWeight / k
    # print("total weight "+ str(totalWeight))
    # print("average wieght" + str(averageWeight))
    currentTruck = []
    for itemIndex in allItems:
        # print("item at index " + str(itemIndex))
        if itemIndex == 0:
            continue
        if currentWeight + allItems[itemIndex] < capaciteMax:

            if currentWeight < averageWeight:
                # print("current item "+ str(allItems[itemIndex]))
                currentTruck.append(itemIndex)
                currentWeight += allItems[itemIndex]
            else:
                # print("reached "+ str(currentWeight))
                currentTruck.append(itemIndex)
                currentWeight += allItems[itemIndex]
                # print("adding truck "+ str(currentTruck)+ " with weight "+ str(currentWeight))
                currentWeight = 0
                splitList.append(currentTruck.copy())
                currentTruck.clear()
                # print("1st current truck  " + str(currentTruck))
        else:
            # print("reached "+ str(currentWeight)+ ", next item "+ str(allItems[itemIndex]) + "would overload")
            # print("adding truck "+ str(currentTruck)+ " with weight "+ str(currentWeight))
            currentWeight = 0
            splitList.append(currentTruck.copy())
            currentTruck.clear()
            currentTruck.append(itemIndex)
            currentWeight += allItems[itemIndex]
            # print("2nd current truck " + str(currentTruck))

    if currentWeight > 0:
        # print("adding last truck "+ str(currentTruck)+ " with weight "+ str(currentWeight))
        splitList.append(currentTruck)

    return splitList


# allItems = {0:0,1:5,2:18,3:6,4:15,5:20,6:8,7:15,8:20,9:13}
#allItems = {0: 0, 1: 24, 2: 18, 3: 30, 4: 45, 5: 20, 6: 12, 7: 15, 8: 20, 9: 13}  # , 10:21,11:25}

#test = splitClients(allItems, 7, 50)
#print("test" + str(test))


# from parse import parse
# from math import hypot

# fileName = "input.txt"


# def parseFile(fileName):
#     instance = {
#         'nodes': list()
#     }

#     citiesCount = None

#     with open(fileName, 'rt') as myFile:
#         python_ne_sait_pas_faire_des_iterators_section = 0

#         for line in myFile.readlines():
#             if line.startswith("NAME : "):
#                 instance['name'] = parse('NAME : {}', line)[0]
#             elif line.startswith("COMMENT : "):
#                 temp, instance['trucksCount'], instance['optimalValue'] = parse(
#                     'COMMENT : {} No of trucks: {}, Optimal value: {})', line)
#             elif line.startswith("TYPE : "):
#                 continue
#             elif line.startswith("DIMENSION : "):
#                 citiesCount = int(parse('DIMENSION : {}', line)[0])
#             elif line.startswith("EDGE_WEIGHT_TYPE : "):
#                 continue
#             elif line.startswith("CAPACITY : "):
#                 instance['trucksCapacity'] = int(parse('CAPACITY : {}', line)[0])
#             elif line.startswith("NODE_COORD_SECTION "):
#                 python_ne_sait_pas_faire_des_iterators_section = 1
#                 continue
#             elif line.startswith("DEMAND_SECTION "):
#                 python_ne_sait_pas_faire_des_iterators_section = 2
#                 continue
#             elif line.startswith("DEPOT_SECTION "):
#                 python_ne_sait_pas_faire_des_iterators_section = 3
#                 continue

#             # ni des switch...
#             if python_ne_sait_pas_faire_des_iterators_section == 1:
#                 id, x, y = parse(' {} {} {}', line)
#                 instance['nodes'].append({'id': int(id) - 1, 'x': int(x), 'y': int(y)})
#             elif python_ne_sait_pas_faire_des_iterators_section == 2:
#                 id, demand = parse('{} {}', line)
#                 instance['nodes'][int(id) - 1]['demand'] = demand[:-1]
#             elif python_ne_sait_pas_faire_des_iterators_section == 3:
#                 instance['depotNodeId'] = int(parse(' {}', line)[0])
#                 break;

#     if citiesCount is None:
#         raise Exception('CitiesCount cannot be None, verify that the input file as a valid format')

#     instance['matrix'] = [[0] * citiesCount for i in range(citiesCount)]
#     # print(citiesCount)

#     for fromNode in instance['nodes']:
#         for toNode in instance['nodes'][fromNode['id']:citiesCount]:
#             dist = int(hypot(toNode['x'] - fromNode['x'], toNode['y'] - fromNode['y']))
#             instance['matrix'][fromNode['id']][toNode['id']] = dist
#             instance['matrix'][toNode['id']][fromNode['id']] = dist

#     # print(instance)
#     # print("")
#     matrix = instance['matrix']
#     truckCapacity = int(instance['trucksCapacity'])
#     truckCount = int(instance['trucksCount'])
#     # print("")
#     # print(instance['nodes'])

#     weights = {}

#     for demand in instance['nodes']:
#         weights[demand.get('id')] = int(demand.get('demand'))

#     # print(weights)

#     print(matrix)

#     return weights, matrix, truckCount, truckCapacity


#parseFile(fileName)





#fileName = "input1.txt"
#allItems, myMatrice, truckCount, truckCapacity = parseFile(fileName)


#myMatrice = matrice2
#allItems = {0:0,1:24,2:18,3:30,4:45,5:20,6:12,7:15,8:20,9:13}#, 10:21,11:25}
#truckCapacity = 50
#truckCount = 5


#paths, results, allResults = recuitCVRPGraph(myMatrice, allItems, truckCount, truckCapacity, 10, 0.95, 100000)


#paths, myresults, allmyresults = recuitCVRPGraph(myMatrice, allItems, truckCount, truckCapacity, 10, 0.95, 10000)
#current_optimum, mean, myresults, allmyresults = recuitCVRPGraph(myMatrice,allItems,truckCount,truckCapacity,10,0.95,10000)

#print(current_optimum)
#print(mean)


#print("done")


# plt.subplot(121)
# plt.plot(range(len(myresults)),myresults)
# #plt.plot(range(len(myresults)),myresults)
# '''
# print("idc")
# G = matriceToGraph(myMatrice)
# labels2 = {}
# for i in G.nodes():
#     labels2[i]= str(i)
# G = nx.relabel_nodes(G, labels2)
# pos = nx.spring_layout(G)
# fig, ax = plt.subplots(figsize=(7,4))
# '''
# fig, ax = plt.subplots(figsize=(7,4))
# plt.plot(range(len(myresults)),myresults)
# plt.plot(range(len(allmyresults)),allmyresults)


#fig2 = plt.figure()

#ax = fig2.add_subplot(1,1,1)

#update(1)

#ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(paths), interval=5, repeat=False)
#plt.show()
#'''
#return ani
#plotGraph(paths)
