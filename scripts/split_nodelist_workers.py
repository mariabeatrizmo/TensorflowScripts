#!/usr/bin/python

import sys

txt = sys.argv[1]

x = txt.split(",c")


first = 1
for line in x:
		if line[0] != "c":
			line = "c" + line
		xx = line.split("-",1)
		if xx[1][0] == "[":
			clean_str = xx[1].replace("[","")	
			clean_str = clean_str.replace("]","")
			numbers = clean_str.split(",")
			for number in numbers:
				if "-" in number:
					num = number.split("-")
					num_begin = int(num[0])
					num_end = int(num[1])
					while (num_begin != num_end):
						print(xx[0],end="-")
						if num_begin < 10:
							print("00",end="")
							print(str(num_begin) + ":2222", end=" ")
						elif num_begin < 100:
							print("0",end="")
							print(str(num_begin) + ":2222", end=" ")
						else:
							print(str(num_begin) + ":2222", end=" ")
						num_begin = num_begin + 1
					if (num_begin == num_end):
						print(xx[0],end="-")
						if num_begin < 10:
							print("00",end="")
							print(str(num_begin) + ":2222", end=" ")
						elif num_begin < 100:
							print("0",end="")
							print(str(num_begin) + ":2222", end=" ")
						else:
							print(num_begin)
						num_begin = num_begin + 1
				else:
					print(xx[0],end="-")
					print(number + ":2222", end=" ")
		else:
			print(xx[0],end="-")
			print(xx[1] + ":2222",end=" ")
print("")
