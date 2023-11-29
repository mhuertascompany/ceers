'''
Yuanzhe Dong, 29 Nov at PKU
'''


from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"
cat = pd.read_csv(os.path.join(cat_dir,cat_name))
z = cat['zfit_50'].values

image_dir1 = "/scratch/ydong/stamps/demo_F200W"
image_dir2 = "/scratch/ydong/stamps/demo_F200W_added"

pred_path = "results/finetune_tree_result/demo_tree_predictions.csv"
pred = pd.read_csv(pred_path)

id = pred['id_str'].values
count_strong = pred['t4_is_there_a_bar__strong_bar_pred'].values
count_weak = pred['t4_is_there_a_bar__weak_bar_pred'].values
count_none = pred['t4_is_there_a_bar__no_bar_pred'].values

p_strong = count_strong/(count_strong+count_weak+count_none)
p_weak = count_weak/(count_strong+count_weak+count_none)

p_bar = p_strong+p_weak
print(np.sum(p_bar>0.5),np.sum(p_bar>0.4),np.sum(p_bar>0.3))


SIZE = 424
max_bar_image_F200W = Image.new('L', (SIZE*5, SIZE*5))
draw_max = ImageDraw.Draw(max_bar_image_F200W)
font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),size=20)


max_bar_ids = np.where(p_bar>0.5)[0]
i = 0
for bar_id in max_bar_ids:
    if id[bar_id] < 20000:
        image_path = os.path.join(image_dir1, "F200W_%i.jpg"%id[bar_id])
        image = Image.open(image_path)
        max_bar_image_F200W.paste(image, (SIZE*(i%5), SIZE*(i//5)))
        draw_max.text((SIZE*(i%5)+10, SIZE*(i//5)+10), f'z={z[bar_id]:.3f}\np_bar={p_bar[bar_id]:.3f}', font=font, fill=255)
        i += 1
    else:
        # image_path = os.path.join(image_dir2, "F200W_%i_a.jpg"%(id[bar_id]-20000))
        continue
    if i == 25:
        break
    
max_bar_image_F200W.save("bar_estimate/max_bars.jpg")

# plt.hist([p_strong, p_weak, p_bar], bins=np.linspace(0,1,11), histtype='step', label=[r'$p_\mathrm{strong}$',r'$p_\mathrm{weak}$',r'$p_\mathrm{bar}$'], linestyle='dashed')
# plt.legend()
# plt.savefig("bar_estimate/bar_hist_F200W.jpg")