
def process_pda(filepath):
	arr = []
	with open(filepath) as my_file:
		for line in my_file:
			s = line[33:]
			arr.append(s)
			#print(s)
	#rint(arr)

	with open(filepath, "w") as txt_file:
		for line in arr:
			txt_file.write("".join(line) )



def mostCommon(arr, n_opcodes):
	s = set()
	for opcode in arr:
		s.add(opcode)
	#print(len(s))
	#print(len(arr))


	#find N most common opcodes
	opcode_counter = {}
	for opcode in arr:
		if opcode in opcode_counter:
			opcode_counter[opcode] += 1
		else:
			opcode_counter[opcode] = 1

	most_common = sorted(opcode_counter, key = opcode_counter.get, reverse = True)
	n_most_common_opcodes = most_common[:n_opcodes]
	#print(n_most_common_opcodes)


	return n_most_common_opcodes

def processOpcodes_pda(filepath):
	arr = []
	#scan in the string of unprocessed opcodes into an array
	with open(filepath) as my_file:
		for line in my_file:
			arr.append(line[:-1]) #-1 so we remove the "/n character"

	#print(arr)
	#print(len(arr))




	#print(possible_opcodes)
	arr[:] = (value for value in arr if value != "invalid") #filter out the "invalid"
	arr2 = []
	i = 0
	for i in range(len(arr)):
		if " " not in arr[i]:
			arr2.append(arr[i])
			#print(arr[i])
		else:
			#temp = arr[i]
			#space = temp.find(" ")
			#temp2 = temp[0:space]
			#print(temp2)
			arr2.append(arr[i][0:arr[i].find(" ")])
		
	#print(arr2)

	arr3 = []	
	#keep only the N most common opcodes

	keep_only_these_opcodes = mostCommon(arr2, 27)
	for unprocessed_opcode in arr2:
		if unprocessed_opcode in keep_only_these_opcodes:
			arr3.append(unprocessed_opcode)
		else:
			arr3.append("other")


	#print(type(possible_opcodes))

	#print(len(arr))

	#rewrite processed opcodes back to text file
	with open(filepath, "w") as txt_file:
		for line in arr3:
			txt_file.write("".join(line) + "\n")





#if __name__ == "__main__":
#	process("VirusShare_fffb1996a5b7c4c716931af2842712e3.txt")