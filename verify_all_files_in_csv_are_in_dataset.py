#this script will remove any files in
import os

#first have a text file with a list of all the virus file names
all_file_names = os.listdir("/Users/allen/Desktop/v001.5")  + os.listdir("/Users/allen/Desktop/v001.4")  + os.listdir("/Users/allen/Desktop/v001.3")  + os.listdir("/Users/allen/Desktop/v001.2") + os.listdir("/Users/allen/Desktop/v001.1")
#print(len(all_file_names))
#print(all_file_names[0])




	#first open the family text file with all files that we could use to extract, but some of them aren't inside dataset
	#are all the file names in the csv file present in the dataset?
		#if so this wont be necessary
	#check if this is true by just using one family
arr = []
with open('/Users/allen/Desktop/Malware-Research/extract_these_files/vbinject_files.txt') as my_file:
    for line in my_file:

        arr.append(line.rstrip())


#print(len(arr))

not_in_main = 0
for name in arr:
	if name not in all_file_names:
		not_in_main += 1
print(not_in_main)
	