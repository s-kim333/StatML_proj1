from datetime import datetime

import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from node2vec import Node2Vec

#dataset = 'C:\Users\admin\Dropbox\Education\University of Melbourne\Semester 2020-02\COMP90051 Statistical Machine Learning\Assignments\Project 1\comp90051-2020-sem2-proj1\train.txt'
# preprocess the data

def timeStamp():
    dateTimeObj = datetime.now().time()
    return dateTimeObj.strftime("%H:%M:%S")

def exportToCSV(df, filename):
    export_path='outputs/' + filename + '.csv'
    df.to_csv(export_path, index=False, header=True)
    print(timeStamp() + filename + ".csv exported")

def exportToTextFile(filename, des, stats):
    export_path='outputs/' + filename + '.txt'
    file = open(export_path, "a+")
    file.write("\n" + des + " : " + stats)
    file.close()

def exportDFtoText(df, filename):
    export_path = 'outputs/' + filename + '.txt'
    np.savetxt(export_path, df.values, fmt='%d')

def findDegreeOfVetex(G, node):
    print(G.degree[node])

def findDegreeOfNodeList(G, node_list):
    degreeValues = list(G.degree(node_list))
    print(degreeValues)

print(timeStamp() + " 1.Importing CSV")
df = pd.read_csv('outputs/all_links.csv')
#print(df)

print(timeStamp() + " 2.Generating Graph")
G = nx.from_pandas_edgelist(df, source='node_1', target='node_2')

node_list = G.nodes()
print("Total Nodes: " + len(node_list).__str__())
exportToTextFile("stats", "Total Nodes", len(node_list).__str__())

edge_list = G.edges()
print("Total Edges: " + len(edge_list).__str__())
exportToTextFile("stats", "Total Edges", len(edge_list).__str__())

print(timeStamp() + " 3.Generating Adjency List")
#df_ajdl = pd.DataFrame(nx.generate_adjlist(G, delimiter=' '))
#exportToCSV(df_ajdl,"adjency_list")
#print(df_ajdl.head(2))

adj_list_node = []
adj_list_node_path = []
adj_list_node_empty = []
for line in nx.generate_adjlist(G, delimiter='\t'):
    #print(line)
    split_nodes_in_path = line.split("\t")
    node = split_nodes_in_path[0]
    adj_list_node.append(node)
    del split_nodes_in_path[0]
    adj_list_node_path.append(split_nodes_in_path)
    if len(split_nodes_in_path) == 0:
        #print("Array size: " + len(split_nodes_in_path).__str__())
        #print("Empty Node: " + node.__str__())
        adj_list_node_empty.append(node)

    #nodes_in_path.append(split_line)
    #print(nodes_in_path)
df_adj_list = pd.DataFrame({'node': adj_list_node, 'path': adj_list_node_path})
#exportToCSV(df_adj_list,"adjency_list")
#exportDFtoText(df_adj_list,"adjency_list")
print(df_adj_list.head(10))
print("Total Leaf Nodes: " + len(adj_list_node_empty).__str__())
exportToTextFile("stats", "Total Leaf Nodes", len(adj_list_node_empty).__str__())
print(timeStamp() + " End")

#df_zero_neigbhor = df_adj_list['path']==[]
#print(df_zero_neigbhor.shape)

#findDegreeOfVetex(df_graph, '3591113')
#findDegreeOfNodeList(df_graph, node_list)

#adj_G = nx.to_numpy_matrix(G, nodelist=node_list)
#adj_G.shape

#plt.figure(figsize=(12,8))
#nx.draw_networkx(df_graph, with_labels=True)

# create graph
#G = nx.from_pandas_edgelist(df, "node_1", "node_2", create_using=nx.Graph())

# plot graph
#plt.figure(figsize=(10,10))

#pos = nx.random_layout(G, seed=23)
#nx.draw(G, with_labels=False,  pos=pos, node_size=40, alpha=0.6, width=0.7)

#plt.show()

# combine all nodes in a list
#node_list = start_node + second_node

# remove duplicate items from the list
#node_list = list(dict.fromkeys(node_list))
#print(node_list)
#print(len(node_list))
# build adjacency matrix
#adj_G = nx.to_numpy_matrix(G, nodelist=node_list)

#adj_G.shape



