# Preprocess your images
python src/prepare_dataset.py --path_to_dataset ./MeGlass_120x120 --path_to_output ./glassesDataset

# Split your database into train and validation sets
split_folders --ratio 0.9 0.1 --output glassesDataset/output  ./glassesDataset

