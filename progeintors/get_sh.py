import pandas as pd
import illustris_python as il

output_dir = "/u/mhuertas/data/CEERS/"
basePath = '/virgotng/universe/IllustrisTNG/TNG100-1/output/'
fields = ['SubhaloMass', 'SubhaloSFRinRad']
subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=fields)

# Add a new key 'RowNumber' to the subhalos dictionary
subhalos['SHID'] = range(len(subhalos['SubhaloMass']))

# Convert the modified subhalos dictionary to a DataFrame
subhalos_df = pd.DataFrame(subhalos)
mass_msun = subhalos['SubhaloMass'] * 1e10 / 0.704

subhalos_massive = subhalos_df.query('SubhaloMass>1e9*0.704/1e10')

# Save the DataFrame to a CSV file
subhalos_massive.to_csv(output_dir + 'subhalos_data_1e9.csv', index=False)
