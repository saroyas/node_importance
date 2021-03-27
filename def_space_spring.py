import networkx as nx
from operator import add, sub
import numpy as np
import matplotlib.pyplot as plt
from python.SpringRank import SpringRank
import python.tools as tl
import copy
import random



def get_space_ranks(graph, self_edge_weight = 1, with_rounding = False, min_iter = 1000, print_rate = 100, max_iter = 10000):
    #intialise dictionary
    # this creates potential issues - are the orders the same in both results???:
    node_page = {}
    for node in list(graph.nodes()):
        node_page[node] = {}

    #calculating what each nodes page will look like
    i = 0
    while True:
        i = i+1
        change = 0
        for node in list(graph.nodes()):
            #The sum on the weights coming in:
            norm_const = 1/(graph.in_degree(node, weight='weight')+self_edge_weight)
            #setting up the new dictionary
            new_page_draft = {}
            new_page_draft[node] = norm_const

            #for each superior node - node that is predecessor
            for superior in list(graph.predecessors(node)):
                #get the weight of the superiors influence
                edge_weight = graph.get_edge_data(superior, node)["weight"]

                #add weight time its page(dictionary) to the current nodes dictionary
                for entry in node_page[superior].keys():
                    new_page_draft[entry] = new_page_draft.get(entry, 0) + edge_weight*(norm_const)*node_page[superior][entry]

            node_page[node] = new_page_draft

        if i%print_rate == 0:
            print("Currently at iteration ", i)

        if (i>max_iter): # (i>min_iter and abs(change) < 1) or (i>max_iter):
            #print("stopped at iteration:", i)
            break

    ranking = {}
    for node in list(graph.nodes()):
        current_node_page = node_page[node]
        for entry in current_node_page.keys():
            ranking[entry] = ranking.get(entry, 0) + current_node_page[entry]

    return ranking


def get_space_ranks_old(graph, self_edge_weight = 1, with_rounding = False, min_iter = 1000, print_rate = 100, max_iter = 10000):
    #intialise dictionary
    # this creates potential issues - are the orders the same in both results???:
    node_page = {}
    for node in list(graph.nodes()):
        node_page[node] = [0.0] * graph.number_of_nodes()

    #calculating what each nodes page will look like
    i = 0
    while True:
        i = i+1
        change = 0
        for node in list(graph.nodes()):
            norm_const = 1/(graph.in_degree(node, weight='weight')+self_edge_weight)
            new_page_draft = [0.0] * graph.number_of_nodes()
            new_page_draft[int(node)] = norm_const
            for superior in list(graph.predecessors(node)):
                edge_weight = graph.get_edge_data(superior, node)["weight"]
                new_page_draft = list(map(add, new_page_draft, [edge_weight*(norm_const)*x for x in node_page[superior]]))

            #change = change + sum(list(map(sub, node_page[node], new_page_draft)))
            node_page[node] = new_page_draft

        if i%print_rate == 0:
            print("Currently at iteration ", i)

        if (i>max_iter): # (i>min_iter and abs(change) < 1) or (i>max_iter):
            #print("stopped at iteration:", i)
            break

    #now calculating each nodes "page" rank values
    #output is list with node i's value at position i
    ranking = [0.0] * graph.number_of_nodes()
    for node in node_page.keys():
        ranking = list(map(add, ranking, node_page[node]))

    if with_rounding == True:
        ranking = [round(elem, 6) for elem in ranking]

    return ranking


def get_spring_ranks(graph, alpha=0., l0=1., l1=1., with_rounding = False):
    nodes = list(graph.nodes())
    A = nx.to_scipy_sparse_matrix(graph, dtype=float,nodelist=nodes)
    ranking =SpringRank(A,alpha=alpha,l0=l0,l1=l1)
    ranking =tl.shift_rank(ranking)
    if with_rounding == True:
        ranking = [round(elem, 6) for elem in ranking]
    return ranking

def reduce_graph(graph, perc_edge_del=0.2):
    number_edge_delete = round(graph.number_of_nodes() * perc_edge_del)
    edges_removed = random.choices(list(graph.edges), k=number_edge_delete)
    reduced_graph = copy.deepcopy(graph)
    reduced_graph.remove_edges_from(edges_removed)

    return reduced_graph, edges_removed

def perc_correct(removed_edges, ranking_list):
    correct = 0
    incorrect = 0
    for edge in removed_edges:
        node1 = edge[0]
        node2 = edge[1]
        if ranking_list[node1] > ranking_list[node2]:
            correct = correct + 1
        elif ranking_list[node1] == ranking_list[node2]:
            correct = correct + 0.5
        else:
            incorrect = incorrect + 1
    return round(correct/len(removed_edges), 6)