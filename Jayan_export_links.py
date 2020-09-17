import pandas as pd
from datetime import datetime
from tqdm import tqdm
import networkx as nx

def timeStamp():
    dateTimeObj = datetime.now().time()
    return dateTimeObj.strftime("%H:%M:%S")

def exportToCSV(df, filename):
    export_path='outputs/' + filename + '.csv'
    df.to_csv(export_path, index=False, header=True)
    print(timeStamp() + filename + ".csv exported")

with open (r"train.txt") as f:
    connections = f.read().splitlines()
print("Total data lines: " + len(connections).__str__())

print(timeStamp() + " Reading Data")
connected_node = []
for con in tqdm(connections):
    split_string = con.split("\t")
    connected_node.append(split_string)

#assert len(connected_node) == 20000

print(timeStamp() + " Generating Links")
start_node = []
second_node = []
for node_list in tqdm(connected_node):
    start = node_list[0]
    connected = node_list
    del connected[0]
    for nodes in connected:
        #if not nodes:
            #print("No neighbours")
        start_node.append(start)
        second_node.append(nodes)

# Complete graph if start node = end nodes
assert len(start_node) == len(second_node)
print("Total Edges: " + len(start_node).__str__())

df_connections = pd.DataFrame({'node_1': start_node, 'node_2': second_node})
#print(df.head(10))

# Exporting all edges
#exportToCSV(df, "all_links")

# combine all nodes in a list
all_nodes = start_node + second_node

# remove duplicate items from the list
all_nodes = list(dict.fromkeys(all_nodes))
initial_node_count = len(all_nodes)
print("Total Nodes: " + initial_node_count.__str__())

df_connections_temp = df_connections.copy()
# empty list to store removable links
omissible_links_index = []

for i in tqdm(df_connections.index.values):
    # remove a node pair and build a new graph
    G_temp = nx.from_pandas_edgelist(df_connections_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())

    # check there is no spliting of graph and number of nodes is same
    if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
        omissible_links_index.append(i)
        df_connections_temp = df_connections_temp.drop(index=i)

print(len(df_connections_temp))
print(omissible_links_index)
print(len(omissible_links_index))



