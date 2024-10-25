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
                        (merge['p_bar_mean'] > 0.5) & 
                        (merge['p_edgeon_mean'] < 0.5) &
                        (merge['p_feature_mean'] > 0.5) &
                        (merge['MAG_MODEL_F444W'] < 25.5)
                         ]
        
    
    SIZE = 424
    images_per_page = 36  # 6x6 grid of images
    pages = []
    draw_max = None
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='serif')), size=20)

    i = 0
    for bar_id, p_feature, p_bar, zr in zip(bars_hz.id_str.values, bars_hz.p_feature_mean.values, bars_hz.p_bar_mean.values, bars_hz.LP_zfinal):
    # Create a new page if necessary
        if i % images_per_page == 0:
            if i > 0:
                pages.append(max_bar_image)  # Save the previous page
            max_bar_image = Image.new('L', (SIZE*6, SIZE*6))  # New page
            draw_max = ImageDraw.Draw(max_bar_image)
    
        # Paste the image and draw the text
        image_path = os.path.join(image_dir, f"{fname}_%i.jpg" % bar_id)
        image = Image.open(image_path)
        max_bar_image.paste(image, (SIZE*(i % 6), SIZE*(i // 6 % 6)))
        draw_max.text((SIZE*(i % 6) + 10, SIZE*(i // 6 % 6) + 10), f'id={bar_id:.3f}\np_feature={p_feature:.3f}\np_bar={p_bar:.3f}', font=font, fill=255)
        i += 1

    # Add the last page
    if i % images_per_page != 0:
        pages.append(max_bar_image)

# Save all pages to a single PDF file
    pdf_path = os.path.join(outdir, f'bar_{filter}_{zlow}_{zhigh}.pdf')
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])


    # Load the main catalog
#cat_dir = "/n03data/huertas/COSMOS-Web/cats"
#cat_name = "COSMOSWeb_master_v2.0.1-sersic-cgs_LePhare-v2_FlaggedM_morphology_zoobot.csv"

cat_dir = "/n03data/huertas/COSMOS-Web/cats"
cat_name = "COSMOS3.1_merged_catalog_effnet_samples.csv"
#cat_cosmos = Table.read(os.path.join(cat_dir,cat_name), format='fits')
#cat_cosmos = hdu[1].data
#names = [name for name in cat_cosmos.colnames if len(cat_cosmos[name].shape) <= 1]
#cat=cat_cosmos[names].to_pandas()
#cat_cosmos = Table.read(os.path.join(cat_dir, cat_name), format='csv')
#names = [name for name in cat_cosmos.colnames if len(cat_cosmos[name].shape) <= 1]
#cat = cat_cosmos[names].to_pandas()
cat = pd.read_csv(os.path.join(cat_dir, cat_name))

z_bins = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

for zb in z_bins:
    select_stamps_and_plot(cat,zb,'/n03data/huertas/COSMOS-Web/zoobot/stamps/','/n03data/huertas/COSMOS-Web/zoobot/bar_candidates')




