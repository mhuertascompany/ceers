'''
Y. Dong, Oct 18
test opening the CEERS catalog
'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


class_dir = "/scratch/ydong/classifications"
class_name = "jwst-ceers-v0-5-aggregated-class-singlechoicequestionsonly.csv"

cla = pd.read_csv(os.path.join(class_dir,class_name))

mag = cla['mag_select'].values
print(len(mag))
featured = cla['t0_smooth_or_featured__features_or_disk__frac'].values
smooth = cla['t0_smooth_or_featured__smooth__frac'].values
print(np.sum(featured+smooth<0.5))

# plt.scatter(featured,smooth,s=1)
# plt.savefig("test/gz_featured_frac_scatter.png")

featured_group = mag[featured>0.5]
smooth_group = mag[smooth>0.5]
print(len(featured_group),len(smooth_group))

bins = range(17,25)
# Plotting the histogram for group 1
plt.hist(featured_group, bins=bins, align='left', alpha=0.7, edgecolor='black', label='featured')

# Plotting the histogram for group 2
plt.hist(smooth_group, bins=bins, align='right', alpha=0.7, edgecolor='black', label='smooth')

# Configuring the plot
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.legend()

plt.savefig("test/mag_hist.png")
