
python preprocess/prepare_mask_dataset.py

PATH_TO_ALIGNED_DATASET="../mask_aligned_with_margin"
PATH_TO_BEARD_DATASET="../beardDataset"

mkdir "${PATH_TO_ALIGNED_DATASET}/nomask"


find ${PATH_TO_BEARD_DATASET}/beard/ -type f -name "*.jpg" -print0 | xargs -0 shuf -e -n 550 -z | xargs -0 cp -vt ${PATH_TO_ALIGNED_DATASET}/nomask
find ${PATH_TO_BEARD_DATASET}/nobeard/ -type f -name "*.jpg" -print0 | xargs -0 shuf -e -n 550 -z | xargs -0 cp -vt ${PATH_TO_ALIGNED_DATASET}/nomask

split_folders --ratio 0.9 0.1 --output ${PATH_TO_ALIGNED_DATASET}/output  ${PATH_TO_ALIGNED_DATASET}



