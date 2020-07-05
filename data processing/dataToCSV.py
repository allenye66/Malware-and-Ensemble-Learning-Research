
import csv
import os


def get_files_by_file_size(dirname, reverse=True):

	filepaths = []
	for basename in os.listdir(dirname):
		#print(basename)
		if basename != ".DS_Store":
			filename = os.path.join(dirname, basename)
			if os.path.isfile(filename):
				filepaths.append(filename)
			 
		 

	for i in range(len(filepaths)):
		filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

	filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

	for i in range(len(filepaths)):
		filepaths[i] = filepaths[i][0]

	return filepaths

def getData(filename, feature_number):

	opcode_features = []

	opcodes = ['mov', 'push', 'call', 'lea', 'add', 'jae', 'inc', 'cmp', 'sub', 'jmp', 'dec', 'shl', 'pop', 'xchg', 'je', 'jne', 'xor', 'test', 'ret', 'jo', 'imul', 'and', 'in', 'jge', 'outsb', 'fstp', 'sbb', 'adc', 'jp', 'insb', 'other']
	

	#for converting the opcodes into numbers
	encode_dict = {}


	c = 1
	for i in opcodes:
		if i != "other":
			encode_dict[i] = c
			c+=1
	#other = 0 and "null" = -1
	encode_dict.update({'other' : 0} )



	#for counting how much of each opcode
	d = {}
	for o in opcodes:
		d[o] = 0

	f =open(filename,'r')
	i = 0

	for line in f:
		if i < feature_number:
			opcode_features.append(encode_dict[line[:-1]])
		d[line[:-1]] += 1
		i += 1
	#print(d)


	#if the array is too small
	#should not happen now
	if len(opcode_features) != 1000:
		for i in range(1000-len(opcode_features)):
			opcode_features.append(-1)

	mov = d['mov']
	push = d['push']
	call = d['call']
	lea = d['lea']
	add = d['add']
	jae = d['jae']
	inc = d['inc']
	cmp_ = d['cmp']
	sub = d['sub']
	jmp = d['jmp']
	dec = d['dec']
	shl = d['shl']
	pop = d['pop']
	xchg = d['xchg']
	je = d['je']
	jne = d['jne']
	xor = d['xor']
	test = d['test']
	ret = d['ret']
	jo = d['jo']
	imul = d['imul']
	and_ = d['and']
	in_ = d['in']
	jge = d['jge']
	outsb = d['outsb']
	fstp = d['fstp']
	sbb = d['sbb']
	adc = d['adc']
	jp = d['jp']
	insb = d['insb']
	other = d['other']

	arr = [i, mov, push, call, lea, add, jae, inc, cmp_, sub, jmp, dec, shl, pop, xchg, je, jne, xor, test, ret, jo, imul, and_, in_, jge, outsb, fstp, sbb, adc, jp, insb, other]
	arr = arr + opcode_features
	return arr


def retrieve_opcodes(filename, total_opcodes):
	arr = []
	f =open(filename,'r')
	for line in f:
		arr.append(line[:-1])




if __name__ == '__main__':

	header = ['File Name','Family', 'Total Opcodes', 'mov', 'push', 'call', 'lea', 'add', 'jae', 'inc', 'cmp', 'sub', 'jmp', 'dec', 'shl', 'pop', 'xchg', 'je', 'jne', 'xor', 'test', 'ret', 'jo', 'imul', 'and', 'in', 'jge', 'outsb', 'fstp', 'sbb', 'adc', 'jp', 'insb', 'other']

	'''
	adload: 1220
	bho: 1408
	ceeinject: 1016
	fakerean: 729
	lolyda_bf: 920
	onlinegames: 1173
	renos: 1455
	startpage: 1253
	vb: 600
	vbinject: 1360
	vobfus: 984
	winwebsec: 2069
	zbot: 519

	'''


	n_features = 1000
	#number_of_files = 900

	for i in range(n_features):
		header.append("Opcode: " + str(i))
	with open('/Users/allen/Desktop/Malware-Research/csv/zbot.csv', 'w') as g:
		writer = csv.writer(g)
		writer.writerow(header)
		files = 0

		family = "/Users/allen/Desktop/ExtractedOpcodes2/zbot_opcodes/"

		for filename in get_files_by_file_size(family):
			if filename != ".DS_Store":
				#filename = family + filename
				row =  getData(filename, n_features)
				row.insert(0, 'ZBOT')
				row.insert(0, filename[-47:])


					#print(filename, ":", totalOpcodes(filename))
					#print(row)
				writer.writerow(row)
			files +=1 

			#for every file inside zbot folder:
				#count the opcodes and add a new row