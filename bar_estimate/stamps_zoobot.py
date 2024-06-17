from PIL import Image, ImageDraw, ImageFont
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from astropy.table import Table

# ignore warnings for readability
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import pandas as pd
from astropy.table import Table

import numpy as np
import pandas as pd
import re


def select_stamps_and_plot(merge,zbin,imdir,outdir): 
    zlow=zbin[0]
    zhigh=zbin[1]

    if (zhigh<=1):
        filter='f150w'
        fname='F150W'
    if (zlow>=1)&(zhigh<=3):
        filter='f277w'
        fname='F277W'
    if (zlow>=3):
        filter='f444w'
        fname='F444W'

    image_dir=os.path.join(imdir,filter)

    bars_hz = merge[(merge['LP_zfinal'] > zlow) & 
                        (merge['LP_zfinal'] < zhigh) & 
                        (merge['LP_mass_med_PDF'] > 9) & 
                        (merge['LP_mass_med_PDF'] < 11) & 
                        (merge['p_spiral_mean'] < 0.5) & 
                        (merge['p_clump_mean'] > 0.5) &
                        (merge['p_feature_mean'] > 0.5)]
        
    
    SIZE = 424
    max_bar_image = Image.new('L', (SIZE*6, SIZE*6))
    draw_max = ImageDraw.Draw(max_bar_image)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='serif')),size=20)
    
    i=0
    for bar_id,p_feature,p_bar,zr in zip(bars_hz.id_str.values,bars_hz.p_feature_mean.values,bars_hz.p_clump_mean.values,bars_hz.LP_zfinal):
        image_path = os.path.join(image_dir, f"{fname}_%i.jpg"%bar_id)
        image = Image.open(image_path)
        max_bar_image.paste(image, (SIZE*(i%6), SIZE*(i//6)))
        draw_max.text((SIZE*(i%6)+10, SIZE*(i//6)+10), f'z={zr:.3f}\np_feature={p_feature:.3f}\np_clump={p_bar:.3f}', font=font, fill=255)
        i += 1
    
        if i == 36:
            break
    
    max_bar_image.save(os.path.join(outdir,f'clump_{filter}_{zlow}_{zhigh}.jpg'))


    # Load the main catalog
cat_dir = "/n03data/huertas/COSMOS-Web/cats"
cat_name = "merged_catalog_tiny.csv"
#cat_cosmos = Table.read(os.path.join(cat_dir, cat_name), format='csv')
#names = [name for name in cat_cosmos.colnames if len(cat_cosmos[name].shape) <= 1]
#cat = cat_cosmos[names].to_pandas()
cat = pd.read_csv(os.path.join(cat_dir, cat_name))

z_bins = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

for zb in z_bins:
    select_stamps_and_plot(cat,zb,'/n03data/huertas/COSMOS-Web/zoobot/stamps/','/n03data/huertas/COSMOS-Web/zoobot/clump_candidates')




