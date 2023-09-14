import illustris_python as il
import pandas as pd
from astropy.io import fits
import numpy as np
import pdb

output_dir="/u/mhuertas/data/CEERS/"
basePath = '/virgotng/universe/IllustrisTNG/TNG100-1/output/'



cat = pd.read_csv(output_dir+"subhalos_data.csv")
fields = ['SubhaloMass','SubfindID','SnapNum','SubhaloMassType']


for idn in cat.SHID:
    tree = il.sublink.loadTree(basePath,99,idn,fields=fields,onlyMPB=True)
    pdb.set_trace()
    if True:
        df = pd.DataFrame(list(zip(tree['SnapNum'], tree['SubfindID'], tree['SubhaloMass'],tree['SubhaloMassType'])), 
               columns =['SnapNUm', 'SubfindID','SubhaloMass','SubhaloMstar']) 
        df.to_csv(output_dir+"TNG100projenitors/TNG100_tree_"+str(idn)+".csv")
    else:
        print("Error")  