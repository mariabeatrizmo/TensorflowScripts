import subprocess
import re 


def ip_finder(line):
   list=""
   found=0
   #match = re.findall(r'\d\d?\d?\.\d\d?\d?\.\d\d?\d?\.\d\d?\d?', line)
   match = re.findall(r'\d\d?\d?\.\d\d?\d?\.\d\d?\d?\.\d\d?\d?:?\d?\d?\d?\d?', line)
   if match:
      for element in match:
          if found != 0:
            list = list + ","
          list = list + element
          found+=1
      print(list)


result = subprocess.run(['hostname'], stdout=subprocess.PIPE).stdout.decode('utf-8')
with open(result[0:8] + "_hosts.txt") as f:
    content = f.read()
    ip_finder(content)
