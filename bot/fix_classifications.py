'''
Fix an error in the original GZ CEERS classification file. 
'''

import pandas as pd
import os
import numpy as np

class_dir = "/scratch/ydong/classifications"
class_name = "jwst-ceers-v0-5-aggregated-class-singlechoicequestionsonly.csv"


cla = pd.read_csv(os.path.join(class_dir,class_name))
cla = cla.rename(columns={"t5_is_there_any_spiral_arm_pattern__no__count ":"t5_is_there_any_spiral_arm_pattern__no__count"})
cla.to_csv(os.path.join(class_dir,class_name))