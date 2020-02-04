#!/bin/bash

# This file downloads validation.zip, runs baseline_np.py on it,
# and packs up the resulting file into ZIP files for upload.

set -e

# Check if numpy and PIL are installed.
python -c "import numpy; import PIL"

VALDIR=val
VALDIROUT=valout

# Get validation.zip
if [[ ! -d $VALDIR ]]; then
  echo "Downloading validation.zip to $VALDIR"
  curl -O https://storage.googleapis.com/clic2020_public/pframe_valonly/validation.zip
  mkdir -p $VALDIR
  mv validation.zip $VALDIR
  pushd $VALDIR
  unzip validation.zip
  popd
fi

# Get encoded .baseline.jpg files
python baseline_np.py $VALDIR $VALDIROUT

# Zip up decoder
zip -r decoder.zip decode baseline_np.py pframe_dataset_shared.py

# Zip up files
zip -r valout.zip $VALDIROUT

echo "Upload decoder.zip and valout.zip"
