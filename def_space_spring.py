import networkx as nx
from operator import add, sub
import numpy as np
import matplotlib.pyplot as plt
from python.SpringRank import SpringRank
import python.tools as tl
import copy
import random

def generalised_page_rank(graph, self_loop_weight = 0, alpha = 0.85,
                                   page_size = 0, end_normalise = False, arrow_dir_powerful = False,
                                   min_iter = 1000, print_rate = 100, max_iter = 10000,
                                   cut_off_change = 0.000001):
    fspace = {}
    for node in graph.nodes():
        fspace[node] = 1/(graph.number_of_nodes())

    norm_const = {}
    for node in graph.nodes():
        if arrow_dir_powerful == True:
            node_dir_weight = graph.out_degree(node, weight = "weight")
        else:
            node_dir_weight = graph.in_degree(node, weight = "weight")

        try:
            norm_const[node] = 1/(node_dir_weight + self_loop_weight)
        except:
            norm_const[node] = 0

    i = 0
    while True:
        i = i+1
        change = 0
        for node in graph.nodes():
            sum_normed_follower_fspace = 0
            if arrow_dir_powerful == True:
                for pred in graph.predecessors(node):
                    sum_normed_follower_fspace = sum_normed_follower_fspace + norm_const[pred]*fspace[pred]*graph.get_edge_data(pred, node)["weight"]
            else:
                for succ in graph.successors(node):
                    sum_normed_follower_fspace = sum_normed_follower_fspace + norm_const[succ]*fspace[succ]*graph.get_edge_data(node, succ)["weight"]

            new_fspace_node = page_size + alpha*(sum_normed_follower_fspace) + (1-alpha)/(graph.number_of_nodes())
            change = change + fspace[node] - new_fspace_node
            fspace[node] = new_fspace_node

        if i%print_rate == 0:
            print("Currently finished iteration ", i)
            print("Change in this iteration", change)

        if (i>max_iter) or (i>min_iter and abs(change) < cut_off_change):
            print("stopped at iteration:", i)
            break

    if end_normalise == True:
        space_ranking = {}
        for node in graph.nodes():
            space_ranking[node] = self_loop_weight * norm_const[node] * fspace[node]

        return space_ranking
    else:
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