import re
filename = r"/scratch1/09111/mbbm/new_profiling/hetero_6_monarch/run-0-c199-081.frontera.tacc.utexas.edu-1699896437.log"
dict={}
count = 0

with open(filename) as file:
    for line in file:
        if line.startswith('{"sys_call_name":"open64",') or line.startswith('{"sys_call_name":"open",') :
            line = line[50:]
            #print(line)
            path = re.search('path":["\d\/\w\-\.]*',line)
            if path != None:
               path=path.group(0)
            else :
               print(line)
               break
            path = path[6:]
            #print(path)
            #if path.startswith('"/scratch1/09111/mbbm/100g_tfrecords/train-00002-'):
            #if path.startswith('"/scratch1/09111/mbbm/100g_tfrecords/train-'):
            if path.startswith('"/scratch1/09111/mbbm/open_images/tfrecords/train/train-'):
                #print(line)
                fd = re.search('result":\d*',line).group(0)
                fd = fd[8:]
                #print( fd + ":" + path)
                dict[fd]=path   #f.write( path + "\n" )


        elif line.startswith('{"sys_call_name":"read",') or line.startswith('{"sys_call_name":"pread",') :
          fd = re.search('fd":\d*',line)
          fd=fd.group(0)[4:]
          if fd in dict:
            #print(line)
            #f.write(line) 
            count = count + 1  #f.write( dict[fd] + "\n" )

        elif line.startswith('{"sys_call_name":"close64",') or line.startswith('{"sys_call_name":"close",') :
           fd = re.search('fd":\d*',line)
           fd=fd.group(0)[4:]
           if fd in dict:
              del dict[fd]
              #f.write(line)
           
print(count)

