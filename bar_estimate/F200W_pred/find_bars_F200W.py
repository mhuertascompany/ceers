'''
Yuanzhe Dong, 6 Dec at PKU
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

image_dir = "/scratch/ydong/stamps/demo_F200W"

pred_path = "bar_estimate/F200W_pred/full_cat_predictions_F200W.csv"
pred = pd.read_csv(pred_path)

id = pred['id_str'].values

count_feature = pred['t0_smooth_or_featured__features_or_disk_pred'].values
count_smooth = pred['t0_smooth_or_featured__smooth_pred'].values
count_artifact = pred['t0_smooth_or_featured__star_artifact_or_bad_zoom_pred'].values
p_feature = count_feature/(count_feature+count_smooth+count_artifact)
p_artifact = count_artifact/(count_feature+count_smooth+count_artifact)

count_edgeon = pred['t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk_pred'].values
count_nonedgeon = pred['t2_could_this_be_a_disk_viewed_edgeon__no_something_else_pred'].values
p_edgeon = count_edgeon/(count_edgeon+count_nonedgeon)

count_strong = pred['t4_is_there_a_bar__strong_bar_pred'].values
count_weak = pred['t4_is_there_a_bar__weak_bar_pred'].values
count_none = pred['t4_is_there_a_bar__no_bar_pred'].values
p_strong = count_strong/(count_strong+count_weak+count_none)
p_weak = count_weak/(count_strong+count_weak+count_none)

p_bar = p_strong+p_weak

for p in [0.5, 0.4, 0.3, 0.2, 0.1, 0]:
    disky = (p_feature>0.3)&(p_edgeon<0.5)
    print(p, np.sum(disky))
    print(np.sum(disky&(p_bar>0.5)),np.sum(disky&(p_bar>0.4)),np.sum(disky&(p_bar>0.3)),np.sum(disky&(p_bar>=0)))


# plt.scatter(p_feature, p_artifact, s=1)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xlabel("p_feature")
# plt.ylabel("p_artifact")
# plt.savefig("bar_estimate/F200W_pred/artifact.jpg")

SIZE = 424
max_bar_image_F200W = Image.new('L', (SIZE*6, SIZE*6))
draw_max = ImageDraw.Draw(max_bar_image_F200W)
font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='serif')),size=20)


max_bar_ids = np.where(p_bar<0.2)[0]
i = 0
for bar_id in max_bar_ids:
    image_path = os.path.join(image_dir, "F200W_%i.jpg"%id[bar_id])
    image = Image.open(image_path)
    max_bar_image_F200W.paste(image, (SIZE*(i%6), SIZE*(i//6)))
    draw_max.text((SIZE*(i%6)+10, SIZE*(i//6)+10), f'z={z[bar_id]:.3f}\np_feature={p_feature[bar_id]:.3f}\np_edgeon={p_edgeon[bar_id]:.3f}\np_bar={p_bar[bar_id]:.3f}', font=font, fill=255)
    i += 1
    
    if i == 36:
        break
    
max_bar_image_F200W.save("bar_estimate/F200W_pred/min_bars.jpg")

plt.hist([p_strong, p_weak, p_bar], bins=np.linspace(0,1,11), histtype='step', label=[r'$p_\mathrm{strong}$',r'$p_\mathrm{weak}$',r'$p_\mathrm{bar}$'], linestyle='dashed')
plt.legend()
plt.savefig("bar_estimate/F200W_pred/bar_hist_F200W.jpg")