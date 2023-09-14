import pandas as pd
import illustris_python as il

output_dir="/u/mhuertas/data/CEERS/"

basePath = '/virgotng/universe/IllustrisTNG/TNG100-1/output/'
fields = ['SubfindID','SubhaloMass', 'SubhaloSFRinRad']
subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=fields)

# Convert the subhalos dictionary to a DataFrame
subhalos_df = pd.DataFrame(subhalos)

# Save the DataFrame to a CSV file
subhalos_df.to_csv(output_dir+'subhalos_data.csv', index=False)
