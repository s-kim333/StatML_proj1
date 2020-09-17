import pandas as pd

def exportToCSV(df, filename):
    export_path='outputs/' + filename + '.csv'
    df.to_csv(export_path, index=False, header=True)
    print(filename + ".csv exported")

df_actual_test = pd.read_csv('data/test-public.txt', sep="\t", header=None, names=["Id", "source", "sink"])
df_actual_test = df_actual_test.drop(df_actual_test.index[0])
#df_actual_test = df_actual_test.drop(columns='Id')
#df_actual_test['link']='null'
print(df_actual_test)
exportToCSV(df_actual_test, "test")
