

# Split your database into train and validation sets
split_folders --ratio 0.9 0.1 --output ../split  ../hair_color_database

python preprocessing/prepareDatasets.py


