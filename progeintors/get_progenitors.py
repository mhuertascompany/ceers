import illustris_python as il
import pandas as pd
from astropy.io import fits
import numpy as np
import pdb

output_dir="/u/mhuertas/data/CEERS/"
basePath = '/virgotng/universe/IllustrisTNG/TNG100-1/output/'

Halos99 = il.groupcat.loadHalos(basePath,99,fields=['GroupFirstSub','GroupMass','GroupMassType',])
fields = ['SubhaloMass','SubfindID','SnapNum','SubhaloMassType','SubhaloHalfmassRadType']


for idn,mass in zip(Halos99['GroupFirstSub'],Halos99['GroupMassType'][:,4]):
    if np.log10(mass*1e10/0.704)>9:
        tree = il.sublink.loadTree(basePath,99,idn,fields=fields,onlyMPB=True)
        #pdb.set_trace()
        try:
            df = pd.DataFrame(list(zip(tree['SnapNum'], tree['SubfindID'], np.log10(tree['SubhaloMass']*1e10/0.704),np.log10(tree['SubhaloMassType'][:,4]*1e10/0.704),tree['SubhaloHalfmassRadType'][:,4]/0.704)), 
                columns =['SnapNUm', 'SubfindID','SubhaloMass','SubhaloMstar','SubhaloHalfmassRad']) 
            df.to_csv(output_dir+"TNG100projenitors_sizemass/TNG100_tree_"+str(idn)+".csv")
        except:
            print("Error")  