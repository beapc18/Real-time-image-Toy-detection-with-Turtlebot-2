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
  for file in $path/*.jpg
  do
	if [ $i -le 80 ]
	then 
		rm $file
		i=$((i+1))
	else i=0
	fi
  done
}


if [ "$#" -ne 1 ]; then
showHelp
else
  unwrapVideos "$1"
  echo "file"
fi
