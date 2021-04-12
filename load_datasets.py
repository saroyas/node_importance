#%%

import networkx as nx
from operator import add, sub

import numpy as np
import matplotlib.pyplot as plt
from main_definitions import *
import random
import copy
import python_springrank.tools as tl
import pandas as pd


def set_weights_one(DG):
    for edge in DG.edges():
        DG[edge[0]][edge[1]]['weight'] = 1
    return DG

def multi_to_di(multi_DG):
    DG = nx.DiGraph()
    for u,v,data in multi_DG.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if DG.has_edge(u,v):
            DG[u][v]['weight'] += w
        else:
            DG.add_edge(u, v, weight=w)
    return DG


def load_faculty(filelocation):
    dataframe_network = pd.read_csv(filelocation, sep="\t")
    dataframe_network = dataframe_network.drop("gender", 1)

    def position_to_weight(position):
        if position == "Assoc":
            return 1
        else:
            return 2

    dataframe_network['rank'] = dataframe_network['rank'].map(lambda position: position_to_weight(position))
    dataframe_network = dataframe_network.rename(columns={'rank': 'weight', "# u": "source", 'v': 'target'})
    multi_DG = nx.from_pandas_edgelist(dataframe_network, edge_attr="weight", create_using=nx.MultiGraph())

    return multi_DG

def load_dataset(dataset):
    if dataset == "twitter":
        DG = set_weights_one(nx.read_edgelist("data/twitter_combined.txt",create_using=nx.DiGraph(), nodetype = int))

    elif dataset == "email_EU":
        DG = set_weights_one(nx.read_edgelist("data/email-Eu-core.txt",create_using=nx.DiGraph(), nodetype = int))

    elif dataset == "Corporate Ownership":
        DG = set_weights_one(nx.read_edgelist("data/ownership.txt",create_using=nx.DiGraph(), nodetype = int))

    elif dataset =="Wikipedia Admin Votes":
        DG = set_weights_one(nx.read_edgelist("data/Wiki-Vote.txt",create_using=nx.DiGraph(), nodetype = int))

    elif dataset =="DBLP Citations":
        DG = set_weights_one(nx.read_edgelist("data/dblp-cite/out.dblp-cite",create_using=nx.DiGraph(), nodetype = int, data=False))

    elif dataset =="Faculty Hiring (History)":
        multi_DG = load_faculty("data/faculty_hiring/History_edgelist.txt")
        DG = multi_to_di(multi_DG)

    elif dataset =="Faculty Hiring (CS)":
        multi_DG = load_faculty("data/faculty_hiring/ComputerScience_edgelist.txt")
        DG = multi_to_di(multi_DG)

    elif dataset =="Faculty Hiring (Business)":
        multi_DG = load_faculty("data/faculty_hiring/Business_edgelist.txt")
        DG = multi_to_di(multi_DG)

    elif dataset =="Political Blog Links":
        multi_DG = nx.read_gml("data/polblogs.gml")
        DG = multi_to_di(multi_DG).reverse()

    elif dataset =="C. Elegans Neural":
        multi_DG = nx.read_gml("data/celegansneural.gml")
        DG = multi_to_di(multi_DG)

    elif dataset == "Drosophila Medulla":
        multi_DG = nx.read_graphml("data/drosophila_medulla_1.graphml")
        DG = multi_to_di(multi_DG)

    return DG