import os

directories = ['/Users/allen/Desktop/ExtractedOpcodes2/ceeinject_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/fakerean_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/lolyda_bf_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/onlinegames_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/renos_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/startpage_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/vb_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/vbinject_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/vobfus_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/winwebsec_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/zbot_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/adload_opcodes', '/Users/allen/Desktop/ExtractedOpcodes2/bho_opcodes']

for directory in directories:
	lb = 1000
	files = 0 
	for filename in os.listdir(directory):
		if filename != '.DS_Store':
			filename = directory + '/' + filename
			#print(filename)
			total_lines = 0

			with open(filename) as f:
				for line in f:
					total_lines += 1

			if total_lines > lb:
				files += 1
			total_lines = 0

	print(files)