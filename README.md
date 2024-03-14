# bars

For reproducing the pipeline, make_stamps/demo_F200W.py is for cutting out greyscale images as the first step. 

The bot folder is for matching the catalog with GZ classifications and finetuning the model. Specifically, we use match_catalog.py and train_on_gz_ceers_tree.py. Besides, gz_ceers_schema.py contains the GZ CEERS decision tree, and To3d.py defines a method to address 2d greyscales in Zoobot. 

The bar_estimate folder is for using the finetuned model to make predictions, as well as analyzing the results to find bars. The scripts estimate_full_catalog.py and F200W_pred/find_bars_F200W.py serve this purpose. 