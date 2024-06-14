from PIL import Image, ImageDraw, ImageFont
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from astropy.table import Table, Column
from astropy.io import fits

# ignore warnings for readability
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import pandas as pd
from astropy.table import Table

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)






import numpy as np
import pandas as pd
import re



# Define a function to load and merge a prediction catalog
def merge_with_predictions(cat, pred_path, filter_name):
    pred = pd.read_csv(pred_path)
    pred = pred.add_suffix(f'_{filter_name}')
    pred.rename(columns={f'id_{filter_name}': 'id_str'}, inplace=True)
    merged = cat.merge(pred, how='inner', right_on=f'id_str', left_on='Id', suffixes=(None, f'_{filter_name}'))
    return merged



# Function to select the correct probabilities based on redshift conditions
def select_probabilities(row):
    conditions = [row['LP_zfinal'] < 1, 1 <= row['LP_zfinal'] < 3, row['LP_zfinal'] >= 3]
    p_feature_choices = [row['p_feature_f150w'], row['p_feature_f277w'], row['p_feature_f444w']]
    p_bar_choices = [row['p_bar_f150w'], row['p_bar_f277w'], row['p_bar_f444w']]
    p_edgeon_choices = [row['p_edgeon_f150w'], row['p_edgeon_f277w'], row['p_edgeon_f444w']]
    p_clump_choices = [row['p_clump_f150w'], row['p_clump_f277w'], row['p_clump_f444w']]
    p_spiral_choices = [row['p_spiral_f150w'], row['p_spiral_f277w'], row['p_spiral_f444w']]
    p_merger_choices = [row['p_merger_f150w'], row['p_merger_f277w'], row['p_merger_f444w']]
    #rf_mag_choices = [row['MAG_MODEL_F150W'], row['MAG_MODEL_F277W'] - 0.6, row['MAG_MODEL_F444W'] - 0.5]

    return (
        np.select(conditions, p_feature_choices, default=np.nan),
        np.select(conditions, p_bar_choices, default=np.nan),
        np.select(conditions, p_edgeon_choices, default=np.nan),
        np.select(conditions, p_clump_choices, default=np.nan),
        np.select(conditions, p_spiral_choices, default=np.nan),
        np.select(conditions, p_merger_choices, default=np.nan),
        #np.select(conditions, rf_mag_choices, default=np.nan)  # Return single value for RF_mag_samples
    )

# Function to parse and clean count arrays from string representation
def parse_count_array(count_str):
    # Remove all non-numeric characters except for commas
    clean_str = re.sub(r'[^\d,]', '', count_str)
    # Split the cleaned string by commas
    return np.array([int(x) for x in clean_str.split(',')])

# Vectorized function to calculate probabilities for all samples in a column
def calculate_probabilities_feature(df, count_column, n_vols):
    return df[count_column].apply(lambda x: parse_count_array(x) / n_vols)

def calculate_probabilities_edge(df, count_column, nvols_column):
    return df.apply(lambda row: parse_count_array(row[count_column]) / parse_count_array(row[nvols_column]), axis=1)

def calculate_probabilities_bar(df, count_column, nfeature_column,nedge_column):
    return df.apply(lambda row: parse_count_array(row[count_column]) / (parse_count_array(row[nfeature_column])-parse_count_array(row[nedge_column])), axis=1)

def calculate_probabilities_clump(df, count_column, nfeature_column,nspiral_column):
    
    return df.apply(lambda row: parse_count_array(row[count_column]) / (parse_count_array(row[nfeature_column])-parse_count_array(row[nspiral_column])), axis=1)

def calculate_probabilities_merger(df, count_column, nfeature_column,nspiral_column,nsmooth_column):
    
    return df.apply(lambda row: parse_count_array(row[count_column]) / (parse_count_array(row[nfeature_column])-parse_count_array(row[nspiral_column])+parse_count_array(row[nsmooth_column])), axis=1)


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

    # Ensure the p_bar_samples and p_feature_samples columns are lists/arrays
    merge['p_bar_samples'] = merge['p_bar_samples'].apply(lambda x: np.array(x))
    merge['p_feature_samples'] = merge['p_feature_samples'].apply(lambda x: np.array(x))

    # Calculate the mean of the samples
    merge['p_bar_mean'] = merge['p_bar_samples'].apply(np.mean)
    merge['p_feature_mean'] = merge['p_feature_samples'].apply(np.mean)

    
    bars_hz = merge[(merge['LP_zfinal'] > zlow) & 
                        (merge['LP_zfinal'] < zhigh) & 
                        (merge['LP_mass_med_PDF'] > 10) & 
                        (merge['LP_mass_med_PDF'] < 11) & 
                        (merge['AXRATIO'] > 0.5) & 
                        (merge['p_bar_mean'] > 0.5) & 
                        (merge['p_feature_mean'] > 0.5)]
        
    
    SIZE = 424
    max_bar_image = Image.new('L', (SIZE*6, SIZE*6))
    draw_max = ImageDraw.Draw(max_bar_image)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='serif')),size=20)
    
    i=0
    for bar_id,p_feature,p_bar,zr in zip(bars_hz.id_str.values,bars_hz.p_feature_mean.values,bars_hz.p_bar_mean.values,bars_hz.LP_zfinal):
        image_path = os.path.join(image_dir, f"{fname}_%i.jpg"%bar_id)
        image = Image.open(image_path)
        max_bar_image.paste(image, (SIZE*(i%6), SIZE*(i//6)))
        draw_max.text((SIZE*(i%6)+10, SIZE*(i//6)+10), f'z={zr:.3f}\np_feature={p_feature:.3f}\np_bar={p_bar:.3f}', font=font, fill=255)
        i += 1
    
        if i == 36:
            break
    
    max_bar_image.save(os.path.join(outdir,f'bars_{filter}_{zlow}_{zhigh}.jpg'))
    



# Load the main catalog
cat_dir = "/n03data/huertas/COSMOS-Web/cats"
cat_name = "COSMOSWeb_master_v2.0.1-sersic-cgs_LePhare-v2_FlaggedM.fits"
cat_cosmos = Table.read(os.path.join(cat_dir, cat_name), format='fits')
names = [name for name in cat_cosmos.colnames if len(cat_cosmos[name].shape) <= 1]
cat = cat_cosmos[names].to_pandas()



# Paths to the prediction catalogs - CHANGE PATHS WHEN DOWNLOADED
pred_paths = {
    'f150w': "/n03data/huertas/COSMOS-Web/cats/gzoo_COSMOS_f277w_effnet_m27_sampling.csv",
    'f277w': "/n03data/huertas/COSMOS-Web/cats/gzoo_COSMOS_f277w_effnet_m27_sampling.csv",
    'f444w': "/n03data/huertas/COSMOS-Web/cats/gzoo_COSMOS_f444w_effnet_m27_sampling.csv"
}

# Merge with each prediction catalog
merged_cat = cat.copy()
for filter_name, pred_path in pred_paths.items():
    merged_cat = merge_with_predictions(merged_cat, pred_path, filter_name)

# Remove redundant 'id_str' columns after merging
for filter_name in pred_paths.keys():
    col_name = f'id_str_{filter_name}'
    if col_name in merged_cat.columns:
        merged_cat.drop(columns=[col_name], inplace=True)

# Final merged catalog
merge = merged_cat


N_VOLS = 100
n_samples=100


# Calculate probabilities for each filter and store them in the DataFrame
merge['p_feature_f150w'] = calculate_probabilities_feature(merge, 'feature_count_f150w', np.zeros(n_samples)+N_VOLS)
merge['p_edgeon_f150w'] = calculate_probabilities_edge(merge, 'edgeon_count_f150w','feature_count_f150w')
merge['p_bar_f150w'] = calculate_probabilities_bar(merge, 'bar_count_f150w', 'feature_count_f150w','edgeon_count_f150w')
merge['p_clump_f150w'] = calculate_probabilities_clump(merge, 'clump_count_f150w', 'feature_count_f150w','spiral_count_f150w')
merge['p_spiral_f150w'] = calculate_probabilities_bar(merge, 'spiral_count_f150w', 'feature_count_f150w','edgeon_count_f150w')
merge['p_merger_f150w'] = calculate_probabilities_merger(merge, 'merger_count_f150w', 'feature_count_f150w','spiral_count_f150w','smooth_count_f150w')

merge['p_feature_f277w'] = calculate_probabilities_feature(merge, 'feature_count_f277w', np.zeros(n_samples)+N_VOLS)
merge['p_edgeon_f277w'] = calculate_probabilities_edge(merge, 'edgeon_count_f277w', 'feature_count_f277w')
merge['p_bar_f277w'] = calculate_probabilities_bar(merge, 'bar_count_f277w', 'feature_count_f277w','edgeon_count_f277w')
merge['p_clump_f277w'] = calculate_probabilities_clump(merge, 'clump_count_f277w', 'feature_count_f277w','spiral_count_f277w')
merge['p_spiral_f277w'] = calculate_probabilities_bar(merge, 'spiral_count_f277w', 'feature_count_f277w','edgeon_count_f277w')
merge['p_merger_f277w'] = calculate_probabilities_merger(merge, 'merger_count_f277w', 'feature_count_f277w','spiral_count_f277w','smooth_count_f277w')


merge['p_feature_f444w'] = calculate_probabilities_feature(merge, 'feature_count_f444w', np.zeros(n_samples)+N_VOLS)
merge['p_edgeon_f444w'] = calculate_probabilities_edge(merge, 'edgeon_count_f444w', 'feature_count_f444w')
merge['p_bar_f444w'] = calculate_probabilities_bar(merge, 'bar_count_f444w', 'feature_count_f150w','edgeon_count_f150w')
merge['p_clump_f444w'] = calculate_probabilities_clump(merge, 'clump_count_f444w', 'feature_count_f444w','spiral_count_f444w')
merge['p_spiral_f444w'] = calculate_probabilities_bar(merge, 'spiral_count_f444w', 'feature_count_f444w','edgeon_count_f444w')
merge['p_merger_f444w'] = calculate_probabilities_merger(merge, 'merger_count_f444w', 'feature_count_f444w','spiral_count_f444w','smooth_count_f444w')



# Apply the function to each row and store the results
merge[['p_feature_samples', 'p_bar_samples', 'p_edgeon_samples','p_clump_samples','p_spiral_samples','p_merger_samples']] = merge.apply(
    lambda row: pd.Series(select_probabilities(row)), axis=1
)


# Ensure the p_bar_samples and p_feature_samples columns are lists/arrays
merge['p_bar_samples'] = merge['p_bar_samples'].apply(lambda x: np.array(x))
merge['p_feature_samples'] = merge['p_feature_samples'].apply(lambda x: np.array(x))
merge['p_merger_samples'] = merge['p_merger_samples'].apply(lambda x: np.array(x))
merge['p_spiral_samples'] = merge['p_spiral_samples'].apply(lambda x: np.array(x))
merge['p_clump_samples'] = merge['p_clump_samples'].apply(lambda x: np.array(x))
merge['p_edgeon_samples'] = merge['p_edgeon_samples'].apply(lambda x: np.array(x))

# Calculate the mean of the samples
merge['p_bar_mean'] = merge['p_bar_samples'].apply(np.mean)
merge['p_feature_mean'] = merge['p_feature_samples'].apply(np.mean)
merge['p_spiral_mean'] = merge['p_spiral_samples'].apply(np.mean)
merge['p_clump_mean'] = merge['p_clump_samples'].apply(np.mean)
merge['p_merger_mean'] = merge['p_merger_samples'].apply(np.mean)
merge['p_edgeon_mean'] = merge['p_edgeon_samples'].apply(np.mean)

# Remove the specified columns
columns_to_remove = [
    'p_bar_samples', 
    'p_feature_samples', 
    'p_merger_samples', 
    'p_spiral_samples', 
    'p_clump_samples',
    'p_edgeon_samples'
]

def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

merge = merge.drop(columns=columns_to_remove)

# Optimize data types
merge = optimize_dtypes(merge)

#rf_mag_choices = [row['MAG_MODEL_F150W'], row['MAG_MODEL_F277W'] - 0.6, row['MAG_MODEL_F444W'] - 0.5]
rf_mag = merge.MAG_MODEL_F150W.values
z=merge.LP_zfinal.values
rf_mag[(z>1)&(z<3)]=merge.MAG_MODEL_F277W.values[(z>1)&(z<3)]-0.6
rf_mag[(z>3)]=merge.MAG_MODEL_F444W.values[(z>3)]-0.5

merge['RF_mag']=rf_mag

# Function to fill NaNs and infinities in nested arrays
def fill_invalid_values(arr, fill_value):
    if isinstance(arr, list):
        return [fill_invalid_values(x, fill_value) for x in arr]
    elif isinstance(arr, np.ndarray):
        arr = np.where(np.isnan(arr), fill_value, arr)
        arr = np.where(np.isinf(arr), fill_value, arr)
        return arr
    return arr

# Assuming merge is already defined as a DataFrame

# Handle NaN and infinity values by filling them with appropriate placeholder values
for col in merge.columns:
    if merge[col].dtype.kind in 'i':  # Integer columns
        merge[col] = merge[col].fillna(-999).replace([np.inf, -np.inf], -999)
    elif merge[col].dtype.kind in 'f':  # Float columns
        merge[col] = merge[col].fillna(-999.0).replace([np.inf, -np.inf], -999.0)
    elif isinstance(merge[col].iloc[0], (list, np.ndarray)):  # Nested arrays
        merge[col] = merge[col].apply(lambda x: fill_invalid_values(x, -999.0))

# Save DataFrame to CSV file in chunks
csv_path = os.path.join(cat_dir, 'merged_catalog.csv')

chunk_size = 10000
for start in range(0, merge.shape[0], chunk_size):
    end = min(start + chunk_size, merge.shape[0])
    chunk = merge.iloc[start:end]
    if start == 0:
        chunk.to_csv(csv_path, index=False, mode='w')
    else:
        chunk.to_csv(csv_path, index=False, mode='a', header=False)

print(f"DataFrame saved to CSV file: {csv_path}")

# Now, convert each chunk from the CSV file to a FITS file
def convert_csv_to_fits(csv_path, output_fits_path, chunk_size=10000):
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Convert chunk to Astropy Table
        chunk_table = Table.from_pandas(chunk)
        
        # Create FITS columns
        fits_columns = []
        for colname in chunk_table.colnames:
            col_data = chunk_table[colname]
            if isinstance(col_data[0], (list, np.ndarray)):
                data = np.array([np.array(x) for x in col_data], dtype='object')
                fits_columns.append(fits.Column(name=colname, format='PJ()', array=data))
            else:
                # Determine the format based on the dtype of the column
                if col_data.dtype == np.int32:
                    col_format = 'J'  # 32-bit integer
                elif col_data.dtype == np.float32:
                    col_format = 'E'  # 32-bit float
                elif col_data.dtype == np.int16:
                    col_format = 'I'  # 16-bit integer
                elif col_data.dtype.kind in {'U', 'S'}:  # String columns
                    max_len = max(len(str(x)) for x in col_data)
                    col_format = f'A{max_len}'
                else:
                    raise ValueError(f"Unsupported data type {col_data.dtype} for column {colname}")
                fits_columns.append(fits.Column(name=colname, format=col_format, array=col_data))

        # Create a FITS HDU from the FITS columns
        hdu = fits.BinTableHDU.from_columns(fits.ColDefs(fits_columns))

        # Write to FITS file
        if not os.path.exists(output_fits_path):
            hdu.writeto(output_fits_path, overwrite=True)
        else:
            with fits.open(output_fits_path, mode='append') as hdul:
                hdul.append(hdu)
                hdul.writeto(output_fits_path, overwrite=True)


# Write to FITS file
output_fits_path = os.path.join(cat_dir, 'COSMOSWeb_master_v2.0.1-sersic-cgs_LePhare-v2_FlaggedM_morphology_zoobot.fits')
convert_csv_to_fits(csv_path, output_fits_path, chunk_size=10000)
#hdu.writeto(output_fits_path, overwrite=True)


print(f"File saved to: {output_fits_path}")


#z_bins = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

#for zb in z_bins:
#    select_stamps_and_plot(merge,zb,'/n03data/huertas/COSMOS-Web/zoobot/stamps/','/n03data/huertas/COSMOS-Web/zoobot/bar_candidates')


