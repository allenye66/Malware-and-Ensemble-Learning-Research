import os    

def get_files_by_file_size(dirname, reverse=True):
	""" Return list of file paths in directory sorted by file size """

	# Get list of files
	filepaths = []
	for basename in os.listdir(dirname):
		#print(basename)
		if basename != ".DS_Store":
		    filename = os.path.join(dirname, basename)
		    if os.path.isfile(filename):
		        filepaths.append(filename)
			 
		 

	# Re-populate list with filename, size tuples
	for i in range(len(filepaths)):
		filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

	# Sort list by file size
	# If reverse=True sort from largest to smallest
	# If reverse=False sort from smallest to largest
	filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

	# Re-populate list with just filenames
	for i in range(len(filepaths)):
		filepaths[i] = filepaths[i][0]

	return filepaths
if __name__ == '__main__':
	for i in (get_files_by_file_size("/Users/allen/Desktop/ExtractedOpcodes2/adload_opcodes/")):
		print(i)