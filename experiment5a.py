"""
bin/mahout svd --input matrix --numRows 150 --numCols 9999 --rank 50 --output output  -Dmapred.job.queue.name=dev
"""

import csv

from java.lang import Class,System
from org.apache.hadoop.conf import Configuration
from org.apache.hadoop.fs import FileSystem,Path
from org.apache.hadoop.io import IntWritable,SequenceFile
from org.apache.mahout.math import RandomAccessSparseVector,Vector,VectorWritable

f = open('/Users/spoulson/data/dataRev2/PaperAuthor.csv', 'rb') 

cs = csv.reader(f)
cs.next()

cnt = 0

paper_map = {}#author
author_map = {}#auth
System.setProperty("HADOOP_USER_NAME", "hadoop")
System.setProperty("java.security.krb5.realm", "")
System.setProperty("java.security.krb5.kdc", "")


configuration = Configuration()
configuration.set("fs.default.name", "hdfs://desktop:9000");
fs = FileSystem.get(configuration)
matrixWriter = SequenceFile.Writer(fs, configuration, Path("/user/hadoop/big_matrix"),Class.forName("org.apache.hadoop.io.IntWritable"), Class.forName("org.apache.mahout.math.VectorWritable"))   


max = 0
for row in cs:
    max+=1
    
    author_map.setdefault(row[1], len(author_map))

f.seek(0)
cs.next()

N = len(author_map)

print "vector size = ",N

c = 0

for row in cs:
    if row[0] not in paper_map:        
        if paper_map:
            matrixWriter.append(key, value);
        
        key = IntWritable();
        value = VectorWritable();
        vector = RandomAccessSparseVector(N)
        value.set(vector)
           
        key.set(paper_map.setdefault(row[0], len(paper_map)))
    
    c+= 1
    if c % 10000 == 0: print c,"out of" ,max
    vector.setQuick(author_map[row[1]],1)
        
matrixWriter.close()

print "vector size = ",N
           