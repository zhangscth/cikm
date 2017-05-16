#coding=utf-8
import numpy as np

sum=0


with open("result.csv") as f:
    for line in f.readlines():
        line = line.strip()
        line = float(line)
        sum+=line
    print "average:",sum/2000.
