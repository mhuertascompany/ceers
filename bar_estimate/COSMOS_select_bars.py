'''
Analyze the predicted vote counts (only for preliminary results) and pick out the disky and barred sample. 
'''


from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from astropy.table import Table

# load catalog
cat_dir = "/n03data/huertas/COSMOS-Web/cats"
cat_name = "COSMOSWeb_master_v2.0.1-sersic-cgs_LePhare-v2_FlaggedM.fits"
cat_cosmos = Table.read(os.path.join(cat_dir,cat_name), format='fits')
#cat_cosmos = hdu[1].data
names = [name for name in cat_cosmos.colnames if len(cat_cosmos[name].shape) <= 1]
cat=cat_cosmos[names].to_pandas()
z = cat['LP_zfinal'].values

# directory for images
image_dir = "/n03data/huertas/COSMOS-Web/zoobot"

# load the finetuned Zoobot predictions 
pred_path = "/n03data/huertas/COSMOS-Web/cats/bars_COSMOS_F150W.csv"
pred = pd.read_csv(pred_path)


merge=cat.merge(pred,how='inner',right_on='id_str',left_on='Id',suffixes=(None,'_x'))

print(len(merge))
print(len(pred))

id = merge['id_str'].values

# For the part below, the expected alpha value in the Dirichlet distribution is taken as probability for simplicity
count_feature = merge['t0_smooth_or_featured__features_or_disk_pred'].values
count_smooth = merge['t0_smooth_or_featured__smooth_pred'].values
count_artifact = merge['t0_smooth_or_featured__star_artifact_or_bad_zoom_pred'].values
p_feature = count_feature/(count_feature+count_smooth+count_artifact)
p_artifact = count_artifact/(count_feature+count_smooth+count_artifact)

count_edgeon = merge['t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk_pred'].values
count_nonedgeon = merge['t2_could_this_be_a_disk_viewed_edgeon__no_something_else_pred'].values
p_edgeon = count_edgeon/(count_edgeon+count_nonedgeon)

count_strong = merge['t4_is_there_a_bar__strong_bar_pred'].values
count_weak = merge['t4_is_there_a_bar__weak_bar_pred'].values
count_none = merge['t4_is_there_a_bar__no_bar_pred'].values
p_strong = count_strong/(count_strong+count_weak+count_none)
p_weak = count_weak/(count_strong+count_weak+count_none)

p_bar = p_strong+p_weak

q=merge.AXRATIO
m150 = merge.MAG_MODEL_F150W.values
z = merge['LP_zfinal'].values


# show the histogram of p_bar distribution
for p in [0.5, 0.4, 0.3, 0.2, 0.1, 0]:
    disky = (p_feature>p)&(p_edgeon<0.5)
    print(p, np.sum(disky))
    print(np.sum(disky&(p_bar>0.5)),np.sum(disky&(p_bar>0.4)),np.sum(disky&(p_bar>0.3)),np.sum(disky&(p_bar>=0)))




SIZE = 424
max_bar_image_F200W = Image.new('L', (SIZE*6, SIZE*6))
draw_max = ImageDraw.Draw(max_bar_image_F200W)
font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='serif')),size=20)


max_bar_ids = np.where((z>2) & (m150>24) & (p_bar>0.5) & (p_feature>0.5) & (p_edgeon<0.5) & (q>0.5))[0]

print('selected:', len(max_bar_ids))
i = 0
for bar_id in max_bar_ids:
    image_path = os.path.join(image_dir, "F150W_%i.jpg"%id[bar_id])
    image = Image.open(image_path)
    max_bar_image_F200W.paste(image, (SIZE*(i%6), SIZE*(i//6)))
    draw_max.text((SIZE*(i%6)+10, SIZE*(i//6)+10), f'z={z[bar_id]:.3f}\np_feature={p_feature[bar_id]:.3f}\np_edgeon={p_edgeon[bar_id]:.3f}\np_bar={p_bar[bar_id]:.3f}', font=font, fill=255)
    i += 1
    
    if i == 36:
        break
    
max_bar_image_F200W.save("/n03data/huertas/COSMOS-Web/zoobot/bar_candidates/min_bars.jpg")

plt.hist([p_strong, p_weak, p_bar], bins=np.linspace(0,1,11), histtype='step', label=[r'$p_\mathrm{strong}$',r'$p_\mathrm{weak}$',r'$p_\mathrm{bar}$'], linestyle='dashed')
plt.legend()
plt.savefig("/n03data/huertas/COSMOS-Web/zoobot/bar_candidates/bar_hist_F200W.jpg")