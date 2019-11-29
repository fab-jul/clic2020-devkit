#!/bin/bash

# Usage:
# bash download.sh OUTPUT_DIR
#
# Uses gsutil if available, otherwise wget if available, otherwise curl. One must be available
# Downloads are continued
#

OUTPUT_DIR=$1
if [[ -z $OUTPUT_DIR ]]; then
  echo "USAGE: $0 OUTPUT_DIR"
  exit 1
fi

VIDEO_URLS="$(pwd)/video_urls.txt"
GSUTIL_URL=TODO
PARALLEL_CONNECTIONS=16

function progress () {
    COUNTER=0
    while read LINE; do
        COUNTER=$((COUNTER+1))
        echo -ne "\rDownloading with $PARALLEL_CONNECTIONS connections; $COUNTER/$NUM_FILES; ...${LINE: -30};"
    done
    echo ""
}

function download_gsutil() {
  gsutil rsync $GSUTIL_URL $OUTPUT_DIR
}

function download_wget_or_curl() {
  if [[ ! -f $VIDEO_URLS ]]; then
    echo "Error: $VIDEO_URLS is not a file."
    exit 1
  fi

  # check wget or curl available
  which wget
  if [[ $? == 0 ]]; then
    WGET_AVAILABLE=1
  else
    which curl
    if [[ $? == 1 ]]; then
      echo "Error: Neigher wget nor curl available!"
      exit 1
    fi
    WGET_AVAILABLE=0
  fi

  # Start download
  mkdir -p $OUTPUT_DIR
  NUM_FILES=$(wc -l < $VIDEO_URLS)
  echo $NUM_FILES

  echo "Downloading to $OUTPUT_DIR..."
  pushd $OUTPUT_DIR
  if [[ $WGET_AVAILABLE == 1 ]]; then
    cat $VIDEO_URLS | head -n50 | xargs -t -n 1 -P $PARALLEL_CONNECTIONS -I{} wget -c {} -q 2>&1 | progress
  else
    cat $VIDEO_URLS | head -n50 | xargs -t -n 1 -P $PARALLEL_CONNECTIONS -I{} curl -O {} -s -C - 2>&1 | progress
  fi
  popd
}

function unzip_all() {
  pushd $OUTPUT_DIR
  for f in *.zip; do
    unzip -f $f
  done
  popd
}

which gsutil
if [[ $? == 0 ]]; then
  echo "Found gsutil, using it..."
  download_gsutil
else
  download_wget_or_curl
  unzip_all
fi

echo "Done, validating..."
python pframe_dataset_shared.py --validate $OUTPUT_DIR/frames
