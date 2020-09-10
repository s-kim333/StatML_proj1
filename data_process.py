
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


# preprocess the data
nodes = set()
exist_connections = {}
with open (r"C:\Users\kimmy\Desktop\train.txt") as f:
    connections = f.read().splitlines()
print("done step 1")

connected_node = []
for con in connections:
    split_string = con.split("\t")
    key = split_string[0]
    nodes.add(key)
    del split_string[0]
    exist_connections[key]=split_string
    for elem in split_string:
        nodes.add(elem)

print("done step 2")

del connections


print(len(nodes))


possible_connections = {}

with open(r"C:\Users\kimmy\Desktop\test-public.txt","r") as f:
    connection = f.read().splitlines()
print(connection)
print(len(connection))

del connection[0]

test_dict = {}
for line in connection:
    split_nodes = line.split("\t")
    if split_nodes[1] in test_dict:
        connected_to = test_dict[split_nodes[1]]
        connected_to.append(split_nodes[2])
        test_dict[split_nodes[1]] = connected_to
    else:
        test_dict[split_nodes[1]] = split_nodes[2]

#print(len(test_dict))

i = 0
#print(possible_connections[0])
with open (r"C:\Users\kimmy\Desktop\possible_negative_cases.txt","a") as e:
    for node in nodes:

        if node in exist_connections:
            if node in test_dict:
                i += 1
                already_connected = set(exist_connections[node])
                possible_connections[node] = nodes.difference(already_connected)
                to_connect = list(possible_connections[node])
                if test_dict[node] in possible_connections[node]:
                    possible_connections[node].remove(test_dict[node])
                else:
                    continue
                to_connect = list(possible_connections[node])
                for a in range(len(to_connect)):
                    # strings += "\t" + to_connect[a]
                    e.write(node + "\t" + to_connect[a] + "\n")
                # e.write(node + strings + "\n")
                del already_connected
                print("done line ", i)

            else:
                i += 1
                already_connected = set(exist_connections[node])
                possible_connections[node] = nodes.difference(already_connected)
                to_connect = list(possible_connections[node])
                # print(to_connect)
                # strings = ""
                for a in range(len(to_connect)):
                    # strings += "\t" + to_connect[a]
                    e.write(node + "\t" + to_connect[a] + "\n")
                # e.write(node + strings + "\n")
                del already_connected
                print("done line ", i)

        else:
            if node in test_dict:
                node_new = node
                i += 1
                nodes.remove(node)
                possible_connections[node_new] = nodes
                if test_dict[node] in possible_connections[node]:
                    possible_connections[node].remove(test_dict[node])
                else:
                    continue
                to_connect = list(possible_connections[node_new])
                for a in range(len(to_connect)):
                    e.write(node + "\t" + to_connect[a] + "\n")
                nodes.add(node_new)
                print("done line ", i)
            else:
                node_new = node
                i += 1
                nodes.remove(node)
                possible_connections[node_new] = nodes
                nodes.add(node_new)
                #print("done")
                #str = ""
                to_connect = list(possible_connections[node_new])
                for a in range(len(to_connect)):
                    e.write(node+"\t"+ to_connect[a]+"\n")
                print("done line ", i)
            #e.write(node+str+"\n")
            #print("finished!")

e.close()
    

