import re
filename= r"/scratch1/09111/mbbm/new_profiling/hetero_4_monarch_lim/run-0-c196-092.frontera.tacc.utexas.edu-1699972144.log"
dict={}
count = 0

with open(filename) as file:
    for line in file:
        #[HierarchicalDataPlane] client reading from level 1 file train-0????-of-01024 with offset: ? and size: ?
        if line.startswith('[HierarchicalDataPlane] client reading from level 1 file ') :
         count = count + 1 
           
print(count)
