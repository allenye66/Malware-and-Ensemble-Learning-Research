import pandas as pd
import numpy as np


df = pd.read_csv('/Users/allen/Desktop/Malware-Research/csv/all_data.csv')



#ceeinject = 0, fakerean = 1, lolydabf = 2, onlinegames = 3, renos = 4, startpage = 5, vb = 6, vbinject = 7, vobfus = 8, winwebsec = 9, zbbot = 10
df.Family = df.Family.replace({"CEEINJECT": 0})
df.Family = df.Family.replace({"FAKEREAN": 1})
df.Family = df.Family.replace({"LOLYDA_BF": 2})
df.Family = df.Family.replace({"ONLINEGAMES": 3})
df.Family = df.Family.replace({"RENOS": 4})
df.Family = df.Family.replace({"STARTPAGE": 5})
df.Family = df.Family.replace({"VB": 6})
df.Family = df.Family.replace({"VBINJECT": 7})
df.Family = df.Family.replace({"VOBFUS": 8})
df.Family = df.Family.replace({"WINWEBSEC": 9})
df.Family = df.Family.replace({"ZBOT": 10})

#for i in df.Family:
#	print(type(i))


print(df.head())