import illustris_python as il
import pandas as pd
from astropy.io import fits
import numpy as np

output_dir="/u/mhuertas/data/CEERS/"
basePath = '/virgo/simulations/IllustrisTNG/TNG50-1/output/'



cat = pd.read_csv(output_dir+"subhalos_data.csv")
fields = ['SubhaloMass','SubfindID','SnapNum']


for idn in cat.SHID:
    tree = il.sublink.loadTree(basePath,99,idn,fields=fields,onlyMPB=True)
    try:
        df = pd.DataFrame(list(zip(tree['SnapNum'], tree['SubfindID'])), 
               columns =['SnapNUm', 'SubfindID','SubhaloMass']) 
        df.to_csv(output_dir+"TNG100projenitors/TNG100_tree_"+str(idn)+".csv")
    except:
        print("Error")  