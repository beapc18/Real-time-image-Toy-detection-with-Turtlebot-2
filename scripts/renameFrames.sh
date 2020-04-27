#!/bin/bash

# How to use the script
showHelp () {
  echo "Usage:"
  echo "./unwrapFrames.sh <absolutePath>"
  exit 0
}

# Unwrap videos in frames
unwrapVideos () {
  path=$1
  i=0
  for file in $path/*.mp4
  do
    # Load directory
    f=${file##*/}
    folder=${f%.mp4}
    dir=$path"/"$folder
    # Delete 9 of every 10 frames
    for file2 in $dir/*.jpg
    do
      # echo "${file2}" "${path}/Annotations"
      mv "${file2}" "${path}/Annotations"
    done
  done
}


if [ "$#" -ne 1 ]; then
showHelp
else
  unwrapVideos "$1"
  echo "file"
fi
