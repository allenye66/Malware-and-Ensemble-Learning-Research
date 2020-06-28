
import pandas as pd
df = pd.read_csv("/Users/allen/Desktop/Malware-Research/00001.csv")
arr = df.name.value_counts()
for i in arr:
	print(i)
print(len(arr))

