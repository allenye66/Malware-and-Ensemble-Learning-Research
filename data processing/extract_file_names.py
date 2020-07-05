import pandas as pd
df = pd.read_csv("/Users/allen/Desktop/Malware-Research/00001.csv")
files_to_decode = df[["file_name", "name"]]

'''
name_set = set()
for index, row in files_to_decode.iterrows():
	name_set.add(row['name'])
print(name_set)
'''

number_of_families = 25
families = files_to_decode['name'].value_counts()[:number_of_families].index.tolist()
print((families))



''''
vbinject = 0
winwebsec = 0
renos = 0
onlinegames = 0
bho = 0
startpage = 0
adload = 0
vb = 0
vobfus = 0
ceeinject = 0
'''
vbinject = []
winwebsec = []
renos = []
onlinegames =[]
bho = []
startpage = []
adload = []
vb = []
vobfus = []
ceeinject = []
lolyda_bf = []
fakerean = []
zbot = []


agent = []
wintrim_bx = []
allaple_a = []
cycbot_g = []
vundo = []
toga_rfn = []
rimecud_a = []

obfuscator = []
small = []
injector = []
hotbar = []
bifrose = []



for index, row in files_to_decode.iterrows():
	if row['name'] == "VBInject":
		vbinject.append(row['file_name'])
	if row['name'] == "Winwebsec":
		winwebsec.append(row['file_name'])
	if row['name'] == 'Renos':
		renos.append(row['file_name'])
	if row['name'] == 'OnLineGames':
		onlinegames.append(row['file_name'])
	if row['name'] == 'BHO':
		bho.append(row['file_name'])
	if row['name'] == "Startpage":
		startpage.append(row['file_name'])
	if row['name'] == 'Adload':
		adload.append(row['file_name'])
	if row['name'] == 'VB':
		vb.append(row['file_name'])
	if row['name'] == 'Vobfus':
		vobfus.append(row['file_name'])
	if row['name'] == 'CeeInject':
		ceeinject.append(row['file_name'])
	if row['name'] == 'Lolyda.BF':
		lolyda_bf.append(row['file_name'])
	if row['name'] == 'FakeRean':
		fakerean.append(row['file_name'])
	if row['name'] == 'Zbot':
		zbot.append(row['file_name'])


	if row['name'] == 'Agent':
		agent.append(row['file_name'])
	if row['name'] == 'Wintrim.BX':
		wintrim_bx.append(row['file_name'])
	if row['name'] == 'Allaple.A':
		allaple_a.append(row['file_name'])
	if row['name'] == 'Cycbot.G':
		cycbot_g.append(row['file_name'])
	if row['name'] == 'Vundo':
		vundo.append(row['file_name'])
	if row['name'] == 'Toga!rfn':
		toga_rfn.append(row['file_name'])
	if row['name'] == 'Rimecud.A':
		rimecud_a.append(row['file_name'])


	if row['name'] == 'Obfuscator':
		obfuscator.append(row['file_name'])
	if row['name'] == 'Small':
		small.append(row['file_name'])
	if row['name'] == 'Injector':
		injector.append(row['file_name'])
	if row['name'] == 'Hotbar':
		hotbar.append(row['file_name'])
	if row['name'] == 'Bifrose':
		bifrose.append(row['file_name'])





		#print(row["file_name"])

'''
print("vbinject",vbinject )
print("winwebsec",winwebsec )
print("renos",renos )
print("onlinegames",onlinegames )
print("bho",bho )
print("startpage",startpage )
print("adload",adload )
print("vb",vb )
print("vobfus",vobfus )
print("ceeinject",ceeinject )

'''
print("vbinject",len(vbinject))
print("winwebsec", len(winwebsec))
print("renos", len(renos))
print("onlinegames", len(onlinegames))
print("bho", len(bho))
print("startpage", len(startpage ))
print("adload", len(adload ))
print("vb", len(vb))
print("vobfus", len(vobfus ))
print("ceeinject", len(ceeinject ))
print("lolydabf", len(lolyda_bf))
print("fakerean", len(fakerean ))
print("zbot", len(zbot ))

print("agent", len(agent ))
print("wintrim_bx", len(wintrim_bx))
print("allaple_a", len(allaple_a ))
print("cycbot_g", len(cycbot_g ))
print("vundo", len(vundo))
print("toga_rfn", len(toga_rfn ))
print("rimecud_a", len(rimecud_a ))




print("obfuscator", len(obfuscator ))
print("small", len(small ))
print("injector", len(injector))
print("hotbar", len(hotbar ))
print("bifrose", len(bifrose ))


'''
with open("/Users/allen/Desktop/Malware-Research/extract_these_files/winwebsec_files.txt", "w") as txt_file:
		for line in winwebsec:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/vobfus_files.txt", "w") as txt_file:
		for line in vobfus:
			txt_file.write("".join(line))
			txt_file.write("\n")


with open("/Users/allen/Desktop/Malware-Research/extract_these_files/vbinject_files.txt", "w") as txt_file:
		for line in vbinject:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/vb_files.txt", "w") as txt_file:
		for line in vb:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/startpage_files.txt", "w") as txt_file:
		for line in startpage:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/renos_files.txt", "w") as txt_file:
		for line in renos:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/onlinegames_files.txt", "w") as txt_file:
		for line in onlinegames:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/ceeinject_files.txt", "w") as txt_file:
		for line in ceeinject:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/bho_files.txt", "w") as txt_file:
		for line in bho:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/adload_files.txt", "w") as txt_file:
		for line in adload:
			txt_file.write("".join(line))
			txt_file.write("\n")







with open("/Users/allen/Desktop/Malware-Research/extract_these_files/fakerean_files.txt", "w") as txt_file:
		for line in fakerean:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/lolyda_bf_files.txt", "w") as txt_file:
		for line in lolyda_bf:
			txt_file.write("".join(line))
			txt_file.write("\n")


with open("/Users/allen/Desktop/Malware-Research/extract_these_files/zbot_files.txt", "w") as txt_file:
		for line in zbot:
			txt_file.write("".join(line))
			txt_file.write("\n")
'''



with open("/Users/allen/Desktop/Malware-Research/extract_these_files/agent_files.txt", "w") as txt_file:
		for line in agent:
			txt_file.write("".join(line))
			txt_file.write("\n")


with open("/Users/allen/Desktop/Malware-Research/extract_these_files/wintrim_bx_files.txt", "w") as txt_file:
		for line in wintrim_bx:
			txt_file.write("".join(line))
			txt_file.write("\n")


with open("/Users/allen/Desktop/Malware-Research/extract_these_files/allaple_a_files.txt", "w") as txt_file:
		for line in allaple_a:
			txt_file.write("".join(line))
			txt_file.write("\n")



with open("/Users/allen/Desktop/Malware-Research/extract_these_files/cycbot_g_files.txt", "w") as txt_file:
		for line in cycbot_g:
			txt_file.write("".join(line))
			txt_file.write("\n")


with open("/Users/allen/Desktop/Malware-Research/extract_these_files/vundo_files.txt", "w") as txt_file:
		for line in vundo:
			txt_file.write("".join(line))
			txt_file.write("\n")


with open("/Users/allen/Desktop/Malware-Research/extract_these_files/toga_rfn_files.txt", "w") as txt_file:
		for line in toga_rfn:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/rimecud_a_files.txt", "w") as txt_file:
		for line in rimecud_a:
			txt_file.write("".join(line))
			txt_file.write("\n")













print("obfuscator", len(obfuscator ))
print("small", len(small ))
print("injector", len(injector))
print("hotbar", len(hotbar ))
print("bifrose", len(bifrose ))




with open("/Users/allen/Desktop/Malware-Research/extract_these_files/obfuscator_files.txt", "w") as txt_file:
		for line in obfuscator:
			txt_file.write("".join(line))
			txt_file.write("\n")



with open("/Users/allen/Desktop/Malware-Research/extract_these_files/smallfiles.txt", "w") as txt_file:
		for line in small:
			txt_file.write("".join(line))
			txt_file.write("\n")


with open("/Users/allen/Desktop/Malware-Research/extract_these_files/injector_files.txt", "w") as txt_file:
		for line in injector:
			txt_file.write("".join(line))
			txt_file.write("\n")


with open("/Users/allen/Desktop/Malware-Research/extract_these_files/hotbar_files.txt", "w") as txt_file:
		for line in hotbar:
			txt_file.write("".join(line))
			txt_file.write("\n")

with open("/Users/allen/Desktop/Malware-Research/extract_these_files/bifrose_files.txt", "w") as txt_file:
		for line in bifrose:
			txt_file.write("".join(line))
			txt_file.write("\n")








