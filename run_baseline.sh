#!/bin/bash

set -e

# must be there!
python -c "import numpy; import PIL"

VALDIR=val
VALDIROUT=valout

# get validation
if [[ ! -d $VALDIR ]]; then
  curl -O https://storage.googleapis.com/clic2020_public/pframe_valonly/validation.zip
  mkdir -p $VALDIR
  mv validation.zip $VALDIR
  pushd $VALDIR
  unzip validation.zip
  popd
fi

# get .baseline files
python baseline_np.py $VALDIR $VALDIROUT

# zip up decoder
zip -r decoder.zip decode baseline_np.py pframe_dataset_shared.py

# zip uf files
zip -r valout.zip $VALDIROUT

echo "Upload decoder.zip and valout.zip"
