import pandas as pd
from datetime import datetime

from tqdm import tqdm
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from node2vec import Node2Vec

def timeStamp():
    dateTimeObj = datetime.now().time()
    return dateTimeObj.strftime("%H:%M:%S")

def exportToCSV(df, filename):
    export_path='outputs/' + filename + '.csv'
    df.to_csv(export_path, index=False, header=True)
    print(timeStamp() + " " + filename + ".csv exported")


#creating positive sampling
print(timeStamp() + " Importing Positive Links CSV")
df_positive_import = pd.read_csv('outputs/positive_links.csv')
print("Positive links: " + df_positive_import.size.__str__())
df_positive = df_positive_import[:132175].copy()

#print(df_positive)

#creatin negative sampling
print(timeStamp() + " Importing Negative Links CSV")
df_negative = pd.read_csv('data/negative.csv')
print("Negative links: " + df_negative.size.__str__())
#print(df_negative.shape)
#print(df_negative)

print(timeStamp() + " Join negative and positive dataframes")
df_data = df_positive.append(df_negative, ignore_index=True)
#print("Complete DataSet")
#print(df_data)

print("***** Link Count ****")
print(df_data['link'].value_counts())
print("*********************")

df_x = df_data[['node_1','node_2']]
#print(df_x)
xtrain, xtest, ytrain, ytest = train_test_split(df_x,df_data['link'], test_size=0.3, random_state=35)
lr = LogisticRegression()

lr.fit(xtrain,ytrain)
#print("xtest printout")
#print(xtest)
#print("ytest printout")
#print(ytest)

print("Model Accuracy After Training: ")
predictions_xtest = lr.predict_proba(xtest)
print(roc_auc_score(ytest, predictions_xtest[:,1]))

df_actual_test = pd.read_csv('data/test-public.txt', sep="\t", header=None, names=["Id", "node_1", "node_2"])
df_actual_test = df_actual_test.drop(df_actual_test.index[0])
df_actual_test = df_actual_test.drop(columns='Id')
print("Test Data Size " + df_actual_test.size.__str__())
#print(df_actual_test.head(10))
#print(df_actual_test.tail(10))
#predictions_actual = lr.predict(df_actual_test)
predictions_prob = lr.predict_proba(df_actual_test)
#print("Prediction Actual")
#print(predictions_actual[0])

print(timeStamp() + " Prediction Results")
#print(predictions_prob[0])
df_results = pd.DataFrame(predictions_prob, columns=["0","Predicted"])
df_results = df_results.drop(columns='0')
df_results.insert(0,'Id', range(1, 1 + len(df_results)))
#print(df_results.size)
print(df_results.head(10))
exportToCSV(df_results, "predictions_2")