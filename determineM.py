import dataToCSV
from dataToCSV import countOpcodes
import os
'''
s = set()
for filename in os.listdir('/Users/allen/Desktop/ExtractedOpcodes2/sample'):
	if filename != ".DS_Store":
		#print(filename)
		filename = "/Users/allen/Desktop/ExtractedOpcodes2/sample/" + filename
		s, d = countOpcodes(filename)
#print(s)
print(d)
'''

d = [('mov', 10894), ('push', 8491), ('call', 4757), ('lea', 3884), ('add', 2722), ('jae', 2626), ('inc', 923), ('cmp', 742), ('sub', 698), ('jmp', 617), ('dec', 584), ('shl', 543), ('pop', 394), ('xchg', 367), ('je', 359), ('jne', 308), ('xor', 269), ('test', 235), ('ret', 194), ('jo', 183), ('imul', 165), ('and', 141), ('in', 130), ('jge', 126), ('outsb', 126), ('fstp', 121), ('sbb', 121), ('adc', 107), ('jp', 100), ('insb', 91), ('int3', 91), ('wait', 89), ('ror', 88), ('outsd', 86), ('or', 83), ('jl', 79), ('ja', 76), ('les', 74), ('movsx', 70), ('loop', 66), ('sti', 66), ('fnclex', 66), ('das', 60), ('jbe', 56), ('cli', 55), ('neg', 55), ('insd', 54), ('sahf', 52), ('cdq', 52), ('retf', 51), ('aas', 47), ('cmpsb', 45), ('scasb', 45), ('jle', 40), ('out', 38), ('fmul', 36), ('pushfd', 35), ('daa', 35), ('movsb', 34), ('fild', 33), ('sal', 32), ('fisubr', 32), ('int1', 32), ('lds', 30), ('js', 30), ('cmc', 30), ('stosb', 30), ('popfd', 29), ('leave', 29), ('fist', 27), ('pushal', 26), ('cwde', 26), ('loopne', 26), ('popal', 25), ('rcl', 23), ('fld', 22), ('aaa', 21), ('ficom', 20), ('iretd', 19), ('into', 19), ('jnp', 19), ('jg', 18), ('std', 17), ('fisttp', 14), ('lahf', 13), ('setne', 13), ('jb', 12), ('rol', 12), ('scasd', 9), ('stosd', 8), ('fnstsw', 7), ('bound', 7), ('jecxz', 7), ('lodsd', 7), ('cmpsd', 6), ('idiv', 6), ('lcall', 4), ('jns', 4), ('nop', 2), ('movzx', 2), ('loope', 2), ('not', 2), ('fcomp', 2), ('movsd', 2), ('fadd', 2), ('fdiv', 1), ('enter', 1), ('aam', 1), ('sete', 1), ('fsub', 1), ('fsubr', 1)]
i = 0
print(d)
total_amount = 0
for key, value in d:
	total_amount+= value

m = 32
m_sum = 0 

only_these_opcodes = []
for key, value in d:
	if i < m:
		print(i+1, key)
		i += 1
		only_these_opcodes.append(key)
		m_sum += value
	else:

		break
print("total", total_amount)
print("m", m_sum)
print(len(d))
print((m_sum/float(total_amount)))
print(only_these_opcodes)