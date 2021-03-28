import networkx as nx
from operator import add, sub
import numpy as np
import matplotlib.pyplot as plt
from python.SpringRank import SpringRank
import python.tools as tl
import copy
import random

def get_page_space_ranks(graph, self_loop_weight = 1, with_rounding = False,
                         min_iter = 1000, print_rate = 100, max_iter = 10000,
                         cut_off_change = 1):

    fspace = {}
    for node in graph.nodes():
        fspace[node] = 1

    norm_const = {}
    for node in graph.nodes():
        node_in_weight = graph.in_degree(node, weight = "weight")
        norm_const[node] = 1/(node_in_weight + self_loop_weight)


    #calculating what each nodes page will look like
    i = 0
    while True:
        i = i+1
        change = 0
        for node in graph.nodes():
            sum_normed_succ_fspace = 0
            for successor in graph.successors(node):
                sum_normed_succ_fspace = sum_normed_succ_fspace + norm_const[successor]*fspace[successor]*graph.get_edge_data(node, successor)["weight"]

            new_fspace_node = 1 + sum_normed_succ_fspace
            change = change + fspace[node] - new_fspace_node
            fspace[node] = new_fspace_node

        if i%print_rate == 0:
            print("Currently finished iteration ", i)
            print("Change in this iteration", change)

        if (i>max_iter) or (i>min_iter and abs(change) < cut_off_change):
            print("stopped at iteration:", i)
            break

    return fspace



def get_space_ranks(graph, self_loop_weight = 1, with_rounding = False,
                        min_iter = 1000, print_rate = 100, max_iter = 10000,
                        cut_off_change = 1):

    fspace = {}
    for node in graph.nodes():
        fspace[node] = 1

    norm_const = {}
    for node in graph.nodes():
        node_in_weight = graph.in_degree(node, weight = "weight")
        norm_const[node] = 1/(node_in_weight + self_loop_weight)


    #calculating what each nodes page will look like
    i = 0
    while True:
        i = i+1
        change = 0
        for node in graph.nodes():
            sum_normed_succ_fspace = 0
            for successor in graph.successors(node):
                sum_normed_succ_fspace = sum_normed_succ_fspace + norm_const[successor]*fspace[successor]*graph.get_edge_data(node, successor)["weight"]

            new_fspace_node = 1 + sum_normed_succ_fspace
            change = change + fspace[node] - new_fspace_node
            fspace[node] = new_fspace_node

        if i%print_rate == 0:
            print("Currently finished iteration ", i)
            print("Change in this iteration", change)

        if (i>max_iter) or (i>min_iter and abs(change) < cut_off_change):
            print("stopped at iteration:", i)
            break

    space_ranking = {}
    for node in graph.nodes():
        space_ranking[node] = self_loop_weight * norm_const[node] * fspace[node]

    return space_ranking





def get_space_ranks_dict_old(graph, self_edge_weight = 1, with_rounding = False,
                    min_iter = 1000, print_rate = 100, max_iter = 10000,
                    cut_off_change = 1):
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
            list_superiors = list(graph.predecessors(node))

            for superior in list_superiors:
                #get the weight of the superiors influence
                try:
                    edge_weight = graph.get_edge_data(superior, node)["weight"]
                except:
                    edge_weight = 1

                #add weight time its page(dictionary) to the current nodes dictionary
                for entry in node_page[superior].keys():
                    new_page_draft[entry] = new_page_draft.get(entry, 0) + edge_weight*(norm_const)*node_page[superior][entry]

            for entry in node_page[node].keys():
                change = change + new_page_draft[entry] - node_page[node].get(entry,0)


            node_page[node] = new_page_draft

        if i%print_rate == 0:
            print("Currently finished iteration ", i)
            print("Change in this iteration", change)

        if (i>max_iter) or (i>min_iter and abs(change) < cut_off_change):
            print("stopped at iteration:", i)
            break

    ranking = {}
    for node in list(graph.nodes()):
        current_node_page = node_page[node]
        for entry in current_node_page.keys():
            ranking[entry] = ranking.get(entry, 0) + current_node_page[entry]

    return ranking


def get_space_ranks_list_old(graph, self_edge_weight = 1, with_rounding = False, min_iter = 1000, print_rate = 100, max_iter = 10000):
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