import numpy as np
import matplotlib.pyplot as plt
import random
import math
import networkx as nx
import matplotlib.animation


def matrix_to_graph(matrix):
    """Convert the given matrix into a graph"""
    # create new graph
    my_graph = nx.Graph()
    # add nodes to the graph
    for node in range(len(matrix)):
        my_graph.add_node(node)
    # add the edges and their weights to the graph
    for origin_node in range(len(matrix)):
        for destination_node in range(len(matrix)):
            if origin_node > destination_node:
                my_graph.add_edge(origin_node, destination_node, weight=matrix[origin_node][destination_node])
    return my_graph


def convert_paths_for_graph(all_paths):
    """Since Python is smart with types,
    we have to convert multiple arrays of int into new arrays of strings to be able to use them"""
    new_paths = []
    for iteration in all_paths:
        new_path = []
        for truck in iteration:
            new_truck = []
            for node in truck:
                new_truck.append(str(node))
            new_path.append(new_truck)
        new_paths.append(new_path)
    return new_paths


# these values will be passed to update via le Animation library
# so we can't and them as parameters to the update function but we have to give them a default value here
fig, ax = plt.subplots(figsize=(7, 4))
paths = []
G = None
pos = []


def update(frame_number):
    """Animates the graph of all accepted solutions"""
    # clear the old graph
    ax.clear()
    # convert the array of int into an array of string as the library needs stings
    new_paths = convert_paths_for_graph(paths)
    # get the different trucks' paths
    trucks = new_paths[frame_number]
    # used to color paths differently
    colors = ["red", "blue", "green", "black", "orange", "purple", "yellow", "brown"]
    color_index = 0
    for truck_path in trucks:
        # draw all the cities the truck visits
        visited_cities = nx.draw_networkx_nodes(G, pos=pos, nodelist=truck_path, ax=ax)
        visited_cities.set_edgecolor("black")
        # draw the route the truck took
        nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(truck_path, truck_path)), font_color="white", ax=ax)
        # draw edges and color them
        edge_list = [truck_path[k:k + 2] for k in range(len(truck_path) - 1)]
        drawn_edges = nx.draw_networkx_edges(G, pos=pos, edgelist=edge_list, width=3, ax=ax)
        drawn_edges.set_edgecolor(colors[color_index % len(colors)])
        color_index += 1

    # Scale the plot
    ax.set_title("Frame %d:    " % (frame_number + 1) + str(trucks), fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def get_all_swappable_items(current_paths, depot, all_items, max_capacity):
    """Calculate for each truck, which items can be added without overloading it"""
    all_items_possible = []
    truck_id = 1
    for truck in current_paths:
        truck_weight = 0
        items_possible = []
        for city in truck:
            # skip the depot
            if city == depot:
                continue
            # add the weight of each element
            truck_weight += all_items[city]
        # if the weight once we add a new item is low than the maximum capacity
        if truck_weight < max_capacity:
            # mark the item as available if it's not already in the truck and doesn't overload it
            for item_index in all_items:
                if item_index not in truck:
                    item_weight = all_items.get(item_index)
                    if ((truck_weight + item_weight) <= max_capacity) and item_index != 0:
                        items_possible.append(item_index)
        # add all items available for the truck to the list of items for all trucks
        all_items_possible.append(items_possible)
        truck_id += 1
    return all_items_possible


def swap_between_trucks_CVRP(trucks_routes, depot, all_items, max_capacity):
    """Swap two items between two truck without overloading the receiving one"""
    swappable_item = get_all_swappable_items(trucks_routes, depot, all_items, max_capacity)
    index_truck_to_load = random.randrange(len(trucks_routes))

    # should be a 'while' but we would end in infinite loops in some problems' instances
    # so we try to find a suitable truck at random for Pi times the number of cities before giving up
    for iteration in range(len(allItems) * 3):
        if len(swappable_item[index_truck_to_load]) == 0:
            index_truck_to_load = random.randrange(len(trucks_routes))
        else:
            break
        if iteration == len(allItems) * 3 - 1:
            return trucks_routes
    # get the truck to load
    truck_to_load = np.array(trucks_routes[index_truck_to_load]).tolist()
    # determine at random if we select and insert the item of the first or the last city of the trucks' routes
    insert_index = random.randrange(0, len(truck_to_load) - 1) if len(truck_to_load) > 1 else 0
    extract_index = random.randrange(0, len(swappable_item[index_truck_to_load]) - 1) if len(
        swappable_item[index_truck_to_load]) > 1 else 0

    truck_to_unload = []
    # select the item from the list of item possible to load in TruckToLoad
    item_to_load = swappable_item[index_truck_to_load][extract_index]

    # find truck to extract from in the list of possible items
    for truck in trucks_routes:
        if item_to_load in truck:
            truck_to_unload = np.array(truck).tolist()
    # find the index of the truck to unload to update our current solution
    index_truck_to_un_load = trucks_routes.index(truck_to_unload)
    # swap the item
    truck_to_load.insert(insert_index, item_to_load)
    truck_to_unload.remove(item_to_load)
    # update the current solution with the new value
    trucks_routes[index_truck_to_load] = truck_to_load
    trucks_routes[index_truck_to_un_load] = truck_to_unload
    return trucks_routes


def swap_inside_a_truck(trucks_list, maximum_trucks_capacity, all_items, max_capacity):
    """Swap two items inside the same truck"""
    # select a truck at random in the list
    truck_index = random.randrange(len(trucks_list))
    truck = trucks_list[truck_index]

    # if the truck has less than two items, we can't swap them, so we pick another one
    while len(truck) < 2:
        truck_index = random.randrange(len(trucks_list))
        truck = trucks_list[truck_index]
        print("in while 1")

    # generate two randoms to decide which items to swap
    rand_item1 = random.randrange(len(truck))
    rand_item2 = random.randrange(len(truck))

    # if we picked the same item, we re-roll one of them
    while rand_item1 == rand_item2:
        rand_item2 = random.randrange(len(truck))

    # swap the two items
    truck[rand_item1], truck[rand_item2] = truck[rand_item2], truck[rand_item1]

    # update the truck list with the new truck
    trucks_list[truck_index] = truck
    return trucks_list

def print_error(action_id):
    """Print an error if the action_id doesn't correspond to a know action"""
    print("random out of bound" + str(action_id))


def switch_action(x):
    # print("x "+ str(x))
    switcher = {
        1: swap_between_trucks_CVRP,
        2: swap_inside_a_truck

    }
    # print("xx "+ str(x))
    return switcher.get(x, print_error)


def calculate_distance(new_trucks_paths, distances_matrix, depot):
    """Calculate the total distance traveled by all the trucks"""
    total_sum = 0
    for truck_path in new_trucks_paths:
        path_distance = 0
        if len(truck_path) > 0:
            for city in range(0, len(truck_path)):
                # this if should be useless
                if city == depot:
                    path_distance += distances_matrix[truck_path[city]][depot]
                else:
                    path_distance += distances_matrix[truck_path[city]][truck_path[city - 1]]
            # add distance to the last node back to the depot
            path_distance += distances_matrix[truck_path[len(truck_path) - 1]][depot]
            total_sum += path_distance
    return total_sum


def annealing_CVRP(matrix, all_items, depot, number_of_trucks, trucks_maximum_capacity, temperature, temperature_coefficient, iteration_count):
    trucks = []
    current_optimum = 1000000000
    starting_solution = split_clients(all_items, number_of_trucks, trucks_maximum_capacity)
    current_solution = np.array(starting_solution)
    # convert to array just in case
    current_solution = current_solution.tolist()
    # check if there are enough trucks
    number_of_iterations = 0
    results = []
    all_results = []
    # difference of almost 0 to begin with but not exactly 0
    delta_e_avg = 0.00000000001
    for iteration_number in range(iteration_count):
        random_action = random.randrange(1, 2)
        action = switch_action(random_action)
        current_solution = action(current_solution, depot, all_items, trucks_maximum_capacity)
        # if there are more trucks than we truly need, we add more up to the needed amount
        if len(current_solution) < number_of_trucks:
            for truck in range(0, number_of_trucks - len(current_solution)):
                current_solution.append([])

        total_cost = calculate_distance(current_solution, matrix, 0)

        delta_e = abs(total_cost - current_optimum)
        # objective function is worse
        if (total_cost > current_optimum):
            if (iteration_count == 0): delta_e_avg = delta_e
            # generate probability of acceptance
            p = math.exp(-delta_e / (delta_e_avg * temperature))
            # determine whether to accept worse solution
            if random.random() < p:
                # accept the worse solution
                accept = True
            else:
                # don't accept the worse solution
                accept = False
        else:
            # objective function is lower, we accept it
            accept = True
        if accept:
            # update currently accepted solution
            current_optimum = total_cost
            # add the accepted solution and its value in a list to plot them and the end
            optimal_solution = current_solution.copy()
            optimal_solution_copy = []  # xopti.copy()
            # print("accepted "+ str(xc))

            for truck in optimal_solution:
                truck_copy = np.array(truck.copy()).tolist()
                # add the depot node at the beginning and the ed of the route
                truck_copy.insert(depot, depot)
                truck_copy.insert(len(truck_copy), depot)
                optimal_solution_copy.append(truck_copy)

            trucks.append(optimal_solution_copy)
            # add accepted results to plot them
            results.append(current_optimum)
            # increment number of accepted solutions
            number_of_iterations += 1
            # update delta_e_avg
            delta_e_avg = (delta_e_avg * (number_of_iterations - 1) + delta_e) / number_of_iterations
        all_results.append(total_cost)
        temperature = temperature_coefficient * temperature
    print(current_optimum)
    print(optimal_solution)
    mean_cost = sum(all_results) / len(all_results)
    return current_optimum, mean_cost, trucks, results, all_results


def split_clients(all_items, number_of_trucks, max_capacity):
    split_list = []
    current_weight = 0
    total_weight = 0
    # calculate the total weight
    for item_index in all_items:
        total_weight += all_items[item_index]
    # calculate the average weight
    average_weight = total_weight / number_of_trucks
    current_truck = []
    for item_index in all_items:
        if item_index == 0:
            continue
        if current_weight + all_items[item_index] < max_capacity:
            if current_weight < average_weight:
                current_truck.append(item_index)
                current_weight += all_items[item_index]
            else:
                current_truck.append(item_index)
                current_weight += all_items[item_index]
                current_weight = 0
                split_list.append(current_truck.copy())
                current_truck.clear()
        else:
            current_weight = 0
            split_list.append(current_truck.copy())
            current_truck.clear()
            current_truck.append(item_index)
            current_weight += all_items[item_index]
    if current_weight > 0:
        split_list.append(current_truck)
    return split_list


from parse import parse
from math import hypot

file_name = "input.txt"


def parse_file(file_name):
    instance = {
        'nodes': list()
    }

    cities_count = None

    with open(file_name, 'rt') as myFile:
        iterators_section = 0

        for line in myFile.readlines():
            if line.startswith("NAME : "):
                instance['name'] = parse('NAME : {}', line)[0]
            elif line.startswith("COMMENT : "):
                temp, instance['trucksCount'], instance['optimalValue'] = parse(
                    'COMMENT : {} No of trucks: {}, Optimal value: {})', line)
            elif line.startswith("TYPE : "):
                continue
            elif line.startswith("DIMENSION : "):
                cities_count = int(parse('DIMENSION : {}', line)[0])
            elif line.startswith("EDGE_WEIGHT_TYPE : "):
                continue
            elif line.startswith("CAPACITY : "):
                instance['trucksCapacity'] = int(parse('CAPACITY : {}', line)[0])
            elif line.startswith("NODE_COORD_SECTION "):
                iterators_section = 1
                continue
            elif line.startswith("DEMAND_SECTION "):
                iterators_section = 2
                continue
            elif line.startswith("DEPOT_SECTION "):
                iterators_section = 3
                continue

            # ni des switch...
            if iterators_section == 1:
                id, x, y = parse(' {} {} {}', line)
                instance['nodes'].append({'id': int(id) - 1, 'x': int(x), 'y': int(y)})
            elif iterators_section == 2:
                id, demand = parse('{} {}', line)
                instance['nodes'][int(id) - 1]['demand'] = demand[:-1]
            elif iterators_section == 3:
                instance['depotNodeId'] = int(parse(' {}', line)[0])
                break;

    if cities_count is None:
        raise Exception('CitiesCount cannot be None, verify that the input file as a valid format')

    instance['matrix'] = [[0] * cities_count for i in range(cities_count)]
    # print(citiesCount)

    for fromNode in instance['nodes']:
        for toNode in instance['nodes'][fromNode['id']:cities_count]:
            dist = int(hypot(toNode['x'] - fromNode['x'], toNode['y'] - fromNode['y']))
            instance['matrix'][fromNode['id']][toNode['id']] = dist
            instance['matrix'][toNode['id']][fromNode['id']] = dist

    matrix = instance['matrix']
    truckCapacity = int(instance['trucksCapacity'])
    truckCount = int(instance['trucksCount'])
    depot = int(instance['depotNodeId']) - 1
    weights = {}

    for demand in instance['nodes']:
        weights[demand.get('id')] = int(demand.get('demand'))
    return weights, matrix, depot, truckCount, truckCapacity


file_name = "input1.txt"
allItems, myMatrice, depot, truckCount, truckCapacity = parse_file(file_name)
print(truckCount)

current_optimum, mean, paths, myresults, allmyresults = annealing_CVRP(myMatrice, allItems, depot, truckCount, truckCapacity,
                                                                       10, 0.95, 10000)
print(paths)
print(current_optimum)
print(mean)


plt.subplot(111)
plt.plot(range(len(myresults)), myresults)

G = matrix_to_graph(myMatrice)
labels = {}
for node in G.nodes():
    labels[node] = str(node)
G = nx.relabel_nodes(G, labels)
pos = nx.spring_layout(G)

fig, ax = plt.subplots(figsize=(7, 4))
plt.plot(range(len(myresults)), myresults)
plt.plot(range(len(allmyresults)), allmyresults)

fig2 = plt.figure()

ax = fig2.add_subplot(1, 1, 1)

#update(len(paths) - 1)

ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(paths), interval=100, repeat=False)
plt.show()
