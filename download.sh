#!/bin/bash

# Usage:
# bash download.sh OUTPUT_DIR [--max_videos MAX_VIDEOS] [--validation_only]
#
# Uses gsutil if available, otherwise wget if available, otherwise curl. One must be available
# Downloads are continued
#
# Tested on macOS
#

USAGE="USAGE: $0 OUTPUT_DIR [--max_videos MAX_VIDEOS]"

# Argument Parsing -------------------------------------------------------------

OUTPUT_DIR=$1
shift
if [[ -z $OUTPUT_DIR ]]; then
  echo $USAGE
  exit 1
fi

while [[ $# -gt 0 ]]; do
  ARG="$1"
  case $ARG in
    --max_videos)
      MAX_VIDEOS=$2
      if [[ -z $MAX_VIDEOS ]]; then
        echo "Must give number MAX_VIDEOS:"
        echo $USAGE
        exit 1
      fi
      shift; shift;
      continue
      ;;
    --validation_only)
      VALIDATION_ONLY=1
      shift;
      continue
      ;;
    *)
      echo $USAGE
      exit 1
  esac
done

# this is a bit fragile because macOS doesn't support readlink -f. Oh well.
SCRIPT_DIR=$(dirname "$0")
VIDEO_URLS="$SCRIPT_DIR/pframe_video_urls.txt"
GSUTIL_URL="gs://clic2020_public/pframe_trainvaltest"
PARALLEL_CONNECTIONS=16

# Helper -----------------------------------------------------------------------

function progress () {
    INFO=$1
    NUM_FILES=$2
    COUNTER=0
    while read LINE; do
        COUNTER=$((COUNTER+1))
        echo -ne "\r$INFO; $COUNTER/$NUM_FILES; ...${LINE: -30};"
    done
    echo ""
}

function download_gsutil() {
  mkdir -pv "$OUTPUT_DIR"
  gsutil -m rsync $GSUTIL_URL "$OUTPUT_DIR"
}

function download_wget_or_curl() {
  mkdir -pv "$OUTPUT_DIR"

  # copying so that we can run all commands in $OUTPUTDIR (curl doesn't have a --prefix option)
  cp -v "$VIDEO_URLS" "$OUTPUT_DIR"
  VIDEO_URLS=$(basename "$VIDEO_URLS")

  pushd $OUTPUT_DIR

  # make sure this exists!
  if [[ ! -f "$VIDEO_URLS" ]]; then
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
  echo "Downloading to $OUTPUT_DIR..."
  NUM_FILES=$(wc -l < "$VIDEO_URLS")

  function get_urls() {
    if [[ -n $MAX_VIDEOS ]]; then
      cat "$VIDEO_URLS" | head -n$MAX_VIDEOS
    else
      cat "$VIDEO_URLS"
    fi
  }

  INFO="Downloading with $PARALLEL_CONNECTIONS connections"
  if [[ $WGET_AVAILABLE == 1 ]]; then
    get_urls | xargs -t -n 1 -P $PARALLEL_CONNECTIONS -I{} wget -c {} -q 2>&1 | progress "$INFO" $NUM_FILES
  else
    get_urls | xargs -t -n 1 -P $PARALLEL_CONNECTIONS -I{} curl -O {} -s -C - 2>&1 | progress "$INFO" $NUM_FILES
  fi

  popd
}

function unzip_all() {
  pushd $OUTPUT_DIR
  # TODO(fabian) could add GNU parallel support here to make things faster
  # TODO(fabian) could remove zips after unzipping?
  for f in *.zip; do
    echo "$f"
    unzip -qu $f
  done | progress Unzipping $NUM_FILES
  popd
}

# Main -------------------------------------------------------------------------

which gsutil >/dev/null
# gsutil does not support $MAX_VIDEOS!
if [[ $? == 0 && -z $MAX_VIDEOS ]]; then
  echo "Found gsutil, using it..."
  download_gsutil
else
  download_wget_or_curl
fi

echo "Unzipping..."
unzip_all

echo "Done, validating..."
python "$SCRIPT_DIR/pframe_dataset_shared.py" --validate "$OUTPUT_DIR"

if [[ $? == 0 ]]; then
  echo "All good! You may now want to remove the *.zip files in $OUTPUT_DIR"
fi
