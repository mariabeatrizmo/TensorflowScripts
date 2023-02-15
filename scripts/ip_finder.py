import subprocess
import sys
import re

#result       = subprocess.run(['ip', 'add'], stdout=subprocess.PIPE).stdout.decode('utf-8')
result       = subprocess.run(['ip', 'add'], capture_output=True, text=True).stdout

#infiniband   = re.search(r'link\/infiniband.*\ scope\ global', result)
infiniband    = result[result.find('link/infiniband'):]

#inet         = re.search(r'inet[\d\. ]*', infiniband)
inet          = infiniband[infiniband.find('inet'):]

ip           = re.search(r'\d\d?\d?\.\d\d?\d?\.\d\d?\d?\.\d\d?\d?', inet).group(0)
ip_with_port = ip + ":2222"
print(ip_with_port)
