import illustris_python as il
import pandas as pd
from astropy.io import fits
import numpy as np
import pdb

output_dir="/u/mhuertas/data/CEERS/"
basePath = '/virgotng/universe/IllustrisTNG/TNG100-1/output/'

GroupFirstSub = il.groupcat.loadHalos(basePath,99,fields=['GroupFirstSub'])
fields = ['SubhaloMass','SubfindID','SnapNum','SubhaloMassType']


for idn in GroupFirstSub:
    tree = il.sublink.loadTree(basePath,99,idn,fields=fields,onlyMPB=True)
    #pdb.set_trace()
    try:
        df = pd.DataFrame(list(zip(tree['SnapNum'], tree['SubfindID'], tree['SubhaloMass'],tree['SubhaloMassType'][:,4])), 
               columns =['SnapNUm', 'SubfindID','SubhaloMass','SubhaloMstar']) 
        df.to_csv(output_dir+"TNG100projenitors/TNG100_tree_"+str(idn)+".csv")
    except:
        print("Error")  