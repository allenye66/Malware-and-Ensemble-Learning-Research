'''
def process_pif(filepath):
	arr = []
	with open(filepath) as my_file:
		for line in my_file:
			arr.append(line)
			#if len(line) > 11 and line[12] == '0':
			#	s = line[43:]
			#	arr.append(s)

			#print(s)
	#rint(arr)

	with open(filepath, "w") as txt_file:
		for line in arr:
			txt_file.write("".join(line) )


'''
'''
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
'''
def processOpcodes_pif(filepath):
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
			#print(arr[i])
		else:
			#temp = arr[i]
			#space = temp.find(" ")
			#temp2 = temp[0:space]
			#print(temp2)
			#print(arr[i][0:arr[i].find(" ")])
			arr2.append(arr[i][0:arr[i].find(" ")])
	#print(arr2)
		
	#print(arr2)

	arr3 = []	
	#keep only the N most common opcodes

	#keep_only_these_opcodes = mostCommon(arr2, 26)
	keep_only_these_opcodes = ['mov', 'push', 'call', 'lea', 'add', 'jae', 'inc', 'cmp', 'sub', 'jmp', 'dec', 'shl', 'pop', 'xchg', 'je', 'jne', 'xor', 'test', 'ret', 'jo', 'imul', 'and', 'in', 'jge', 'outsb', 'fstp', 'sbb', 'adc', 'jp', 'insb']
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
#	processOpcodes_pif("/Users/allen/Desktop/malware/v001.4/VirusShare_0a5d32eeeed030c32b37239aa5cdc1d2")