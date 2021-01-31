import os

f = open('adult.test')
outfile = open('newadult.test', 'w') 
for line in f.readlines():
    line = line.rstrip()
    line = line[:-1]
    outfile.write(line+'\n')
f.close()
outfile.close()