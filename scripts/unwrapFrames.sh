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
    # Create new directory for storing frames
    f=${file##*/}
    # folder=${f%.mp4}
    # dir=$path"/"$folder
    # mkdir $dir

    # Unwrap video in subfolder (quality from 1-high to 31-low)
    outFile=$path"Annotations/${i}_%08d.jpg"    
    echo $outFile
    i=$((i+1))
    ffmpeg -i $file -qscale:v 2 $outFile
    echo "File $f has been unwraped successfully"
  done
}


if [ "$#" -ne 1 ]; then
showHelp
else
  unwrapVideos "$1"
  echo "file"
fi
