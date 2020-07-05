import r2pipe
import processOutput_pd
import processOutput_pda
from processOutput_pda import process_pda, processOpcodes_pda
from processOutput_pd import process, processOpcodes
from os import listdir
import os
import processOutput_pif
from processOutput_pif import processOpcodes_pif




def iterate(foldername):
	for f in listdir(foldername):
		print(f)
def extract_write(filename, folder_path):
	r = r2pipe.open(filename)
	##any command that one runs in the r2 shell can be passed to the cmd method
	#print(r.cmd('pd'))
	#print(r.cmd('v'))
	#print(r.cmd)
	#s = r.cmd("pd[dD]")
	#pd > output.txt #for terminal


	#s = r.cmd("pd 400") #find better way to do this

	file_size = os.path.getsize(filename)
	print("file size:", file_size) 

	#s = r.cmd(command) # THIS IS DISASSEMBLE? IS IT CORRECT?

	#s = r.cmd("pda")

	r.cmd("e asm.bytes = 0")
	r.cmd("aaa")
	s = r.cmd("pif@@f")
	#print(s)



	#print(s)


	## r.cmd("pD (filesize)")
		#find filesize with wc -c < "filename"

	##create new text file name
	slash = filename.rfind("/")
	#print(slash)
	txt_file = filename[slash + 1:] + ".txt"
	full_directory = folder_path + "/"
	#xt_file = temp_s + txt_file
	#print(txt_file)
	#print(txt_file)

	f = open(os.path.join(folder_path,txt_file), "w+")
	f.write(s)
	f.close()
	return (str(full_directory + txt_file))
	#return ('"%s"' % str(txt_file))


	#f = open("output.txt", "w") #create new text file and write to it
	#f.write(s)
	#f.close()


	#print((extract_write("/Users/allen/Desktop/v001/VirusShare_fffb1996a5b7c4c716931af2842712e3")))
	#s = ((extract_write("/Users/allen/Desktop/v001.1/VirusShare_fda3e6a6a8378f7cdb1369c0a8cf599d")))
	#s = ((extract_write("/Users/allen/Desktop/v001.5/VirusShare_fffb1996a5b7c4c716931af2842712e3"))) 
	#print(s)

	#process_pda(s)
	#processOpcodes_pda(s)

	#process(s)
	#processOpcodes(s)


	#process_pdf(s)
	#rocessOpcodes_pdf(s)
def main_extract(filename, folder):
	s = extract_write(filename, folder)
	#process_pdf(s)
	processOpcodes_pif(s)


  

if __name__ == "__main__":
	stream = os.popen('find /Users/allen/Desktop/malware')
	filename_to_path = {}
	for line in stream.readlines():
		line = line.rstrip()
		pathes = line.split('/')
		filename_to_path[pathes[-1]] = line
		#print(line)
		#print("%s -> %s" % (pathes[-1], line))

	#x = filename_to_path["VirusShare_8fee86c609f6ffdfacaa69066e3ef262"]
	#print(x)
	count = 0
	sizes = set()
	with open("/Users/allen/Desktop/Malware-Research/extract_these_files/injector_files.txt") as my_file:
		for line in my_file:

			##print(get_key(line, filename_to_path))
			
			##line = "/Users/allen/Desktop/v001.1/" + line.rstrip() + "/" #HOW TO CHECK WHICH FOLDER THIS IS IN
			line = filename_to_path[line.rstrip()]
			file_size = os.path.getsize(line.rstrip())

			
			print(line)
			##print(line)
			main_extract(line, "/Users/allen/Desktop/ExtractedOpcodes2/injector_opcodes/")
			count += 1
			print(count)
	#main_extract("/Users/allen/Desktop/v001.1/VirusShare_8fee86c609f6ffdfacaa69066e3ef262","/Users/allen/Desktop/ExtractedOpcodes/vobfus_opcodes")
