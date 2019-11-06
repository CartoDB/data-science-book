import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

import operator
import random
from copy import deepcopy
from itertools import combinations 
import pickle as pkl

import numpy as np
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import *
from geopy.distance import great_circle
from simanneal import Annealer

import cartoframes
from cartoframes.viz import *
from cartoframes.data import *

import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')


#######################################################
# Solve TSP using Ant Colony Optimization in Python 3 #
# Code Source:                                        #
# https://github.com/ppoffice/ant-colony-tsp          #
#######################################################
class Graph(object):
    def __init__(self, cost_matrix: list, rank: int):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        self.matrix = cost_matrix
        self.rank = rank
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]
        logging.info(f'[Done] Load the Graph')
        

class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int):
        """
        :param ant_count:
        :param generations:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
        """
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve(self, graph: Graph):
        """
        :param graph:
        """
        best_cost = float('inf')
        best_solution = []
        all_costs = []
        for gen in range(self.generations):
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ind, ant in enumerate(ants):
                for i in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                # update pheromone
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
            logging.info(f'[Generation #{gen}] [best cost: {best_cost}]')
        return best_solution, best_cost
    
class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(graph.rank)]  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # heuristic information
        start = random.randint(0, graph.rank - 1)  # start from any node
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][i] ** self.colony.beta
        probabilities = [0 for i in range(self.graph.rank)]  # probabilities for moving to a node in the next step
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass  # do nothing
        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost

def distance_aco(cities: dict):
    distance_matrix = []
    for ka, va in cities.items():
        record_distance = []
        for kb, vb in cities.items():
            if kb == ka:
                record_distance.append(0.0)
            else:
                record_distance.append(great_circle(va, vb).m)
        distance_matrix.append(record_distance)
    logging.info(f'[Done] Create A Distance Matrix For ACO')
    return distance_matrix



def location(data: pd.DataFrame, 
             id_col: str, 
             geometry_col: str) -> dict :
    """
    Extract Location from dataframe. Output a dict as {id: (lng, lat), ...}
    """
    loc = {}
    for row in data.iterrows():
        loc_id = row[1][id_col]
        x, y = row[1][geometry_col].x, row[1][geometry_col].y
        loc[loc_id] = loc.get(loc_id, (x,y))
    logging.info(f'[Done] Transform DataFrame To Location Dict)')
    return loc


#######################################################
# Christofides algorithm                              #
# Code Source:                                        #
# https://github.com/Retsediv/ChristofidesAlgorithm   #
#######################################################

def christofides(data):
    # build a graph
    G = build_graph(data)
    # print("Graph: ", G)

    # build a minimum spanning tree
    MSTree = minimum_spanning_tree(G)
    MSTree_init = deepcopy(MSTree)
    # print("MSTree: ", MSTree)

    # find odd vertexes
    odd_vertexes = find_odd_vertexes(MSTree)
    odd_vertexes_init = deepcopy(odd_vertexes)
    # print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    new_added_matching = minimum_weight_matching(MSTree, G, odd_vertexes)
    united_MSTree_perfect_matching = deepcopy(MSTree)
    # print("Minimum weight matching: ", MSTree)

    # find an eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G)

    # print("Eulerian tour: ", eulerian_tour)

    current = eulerian_tour[0]
    path = [current]
    visited = [False] * len(eulerian_tour)

    length = 0

    for v in eulerian_tour[1:]:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]
            current = v

    # path.append(path[0])

    # print("Result path: ", path)
    # print("Result length of the path: ", length)

    return G, MSTree_init, odd_vertexes_init, new_added_matching, united_MSTree_perfect_matching, eulerian_tour, length, path



def get_length(x1, y1, x2, y2, name='great_circle'):
    '''
    x1: lat1
    y1: lng1
    x2: lat2
    y2: lng2
    '''
    if name == 'great_circle':
        return great_circle((x1,y1), (x2,y2)).km
    else:
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


def build_graph(data):
    graph = {}
    for this in range(len(data)):
        for another_point in range(len(data)):
            if this != another_point:
                if this not in graph:
                    graph[this] = {}

                graph[this][another_point] = get_length(data[this][0], 
                                                        data[this][1], 
                                                        data[another_point][0],
                                                        data[another_point][1],
                                                        name='great_circle')

    return graph


class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)

    return tree


def find_odd_vertexes(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)

    return vertexes


def minimum_weight_matching(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)

    new_added = []
    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u

        MST.append((v, closest, length))
        new_added.append((v, closest, length))
        odd_vert.remove(closest)
    return new_added

def find_eulerian_tour(MatchedMSTree, G):
    # find neigbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # print("Neighbours: ", neighbours)

    # finds the hamiltonian circuit
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w

    return EP


def remove_edge_from_matchedMST(MatchedMST, v1, v2):

    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]

    return MatchedMST


def Euler_Tour(multigraph):
	""" Uses Fleury's algorithm to find the Euler Tour of the MultiGraph.
	"""
	tour = []
	temp_graph = nx.MultiGraph()
	graph_nodes = nx.nodes(multigraph)
	current_node = graph_nodes[0]
	tour.append(current_node)
	while nx.number_of_edges(multigraph) > 0: 	
		for edge in multigraph.edges(current_node):
			temp_graph = copy.deepcopy(multigraph)
			temp_graph.remove_edge(edge[0], edge[1], key=None)
			if nx.is_connected(temp_graph):
				tour.append(edge[1])
				current_node = edge[1]
				multigraph.remove_edge(edge[0], edge[1], key=None)
				break
		else:
			tour.append(edge[1])
			current_node = edge[1]
			multigraph.remove_edge(edge[0], edge[1], key=None)
			multigraph.remove_nodes_from(nx.isolates(multigraph))
	return tour


def shortcut_Euler_Tour(tour):
	"""Find's the shortcut of the Euler Tour to obtain the Approximation.
	"""
	Tour = []
	for vertex in tour:
		if vertex not in Tour:
			Tour.append(vertex)
	Tour.append(tour[0])
	return Tour


class TravelingSalesman(Annealer):
    """Calculates sequence of places to visit"""
    def __init__(self, state, distance_matrix):
        self.distance_matrix = distance_matrix
        super(TravelingSalesman, self).__init__(state)

    def move(self):
        """Swaps two cities in the route."""
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """Calculates energy with current configuration"""
        total_dist = 0
        # add distances from i-1 -> i
        for i in range(len(self.state)):
            # loop, back to the start.
            total_dist += self.distance_matrix[self.state[i-1]][self.state[i]]
        return total_dist
    
    
def TravelingSalesmanRun(loc: dict, iteration: int):
    output = pd.DataFrame({'id': [], 'iteration': [], 'distance': []})
    
    # create a distance matrix
    distance_matrix = {}
    for ka, va in loc.items():
        distance_matrix[ka] = {}
        for kb, vb in loc.items():
            if kb == ka:
                distance_matrix[ka][kb] = 0.0
            else:
                distance_matrix[ka][kb] = great_circle(va, vb).m
    logging.info(f'[Done] Create A Distance Matrix)')

    for iter_i in range(iteration):
        # initial state
        
        # init_state = sorted(list(loc.keys()))
        init_state = list(loc.keys())
        random.shuffle(init_state)
        
        # run
        distance_matrix_copy = distance_matrix.copy()
        
        tsp = TravelingSalesman(init_state, distance_matrix_copy)
        
        ##################################################
        # Tmax = 25000.0  # Max (starting) temperature   #
        # Tmin = 2.5      # Min (ending) temperature     #
        # steps = 50000   # Number of iterations         #
        # updates = 100                                  #
        ##################################################
        auto_schedule = tsp.auto(minutes=1) 
        tsp.set_schedule(auto_schedule)
        tsp.copy_strategy = "slice"
        
        state, e = tsp.anneal()
        
        logging.info(f'[{iter_i+1}]: {e} m route)')
        
        # record
        output_i = pd.DataFrame({'id': state, 'iteration': [iter_i]*len(loc), 'distance': [e]*len(loc)})
        output = output.append(output_i)
    logging.info(f'[Done]: Traveling Salesman Run')
    return output

def result(loc: dict, Output: pd.DataFrame) -> pd.DataFrame:
    output_copy = Output.copy()
    loc_copy = loc.copy()
    
    loc_copy = pd.DataFrame.from_dict(loc_copy, orient='index')
    loc_copy.columns = ['lng','lat']
    
    output_copy['shortest'] = output_copy['distance'].rank(method="min")
    output_copy['visitOrder'] = output_copy.index + 1
    output_copy.index = output_copy['id']
    output_copy = output_copy.join(loc_copy)
    output_copy['shortest'] = output_copy['shortest'].astype(int)
    output_copy = output_copy.sort_values(by=['shortest', 'visitOrder'])
    output_copy.reset_index(inplace=True, drop=True)
    output_copy['geometry'] = output_copy.apply(lambda x: Point(x.lng, x.lat), axis=1)
    logging.info(f'[Done]: Organize Result')
    return output_copy

def shortestRoute(result: pd.DataFrame):
    shortest_route = result[result.shortest == 1]
    shortest_route = gpd.GeoDataFrame(shortest_route) 
    logging.info(f'[Done]: Find The Shortest Route')
    return shortest_route

    
