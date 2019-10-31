#!/bin/bash

source_folder="raw_videos/brest/"
name="brest_20190609_130424_327_334"

source_file=${source_folder}/${name}.ts
size_s=900 # 15

start_s=0
end_s=$((start_s+size_s))
video_name_temp=${name}_${start_s}_${end_s};
folder=${source_folder}/${name}/${video_name_temp}
mkdir -p ${folder};
dest_file=${folder}/${video_name_temp}.mp4
ffmpeg -i $source_file -ss $start_s -t $size_s $dest_file

start_s=900
end_s=$((start_s+size_s))
video_name_temp=${name}_${start_s}_${end_s};
folder=${source_folder}/${name}/${video_name_temp}
mkdir -p ${folder};
dest_file=${folder}/${video_name_temp}.mp4
ffmpeg -i $source_file -ss $start_s -t $size_s $dest_file

start_s=1800
end_s=$((start_s+size_s))
video_name_temp=${name}_${start_s}_${end_s};
folder=${source_folder}/${name}/${video_name_temp}
mkdir -p ${folder};
dest_file=${folder}/${video_name_temp}.mp4
ffmpeg -i $source_file -ss $start_s -t $size_s $dest_file


start_s=2700
end_s=$((start_s+size_s))
video_name_temp=${name}_${start_s}_${end_s};
folder=${source_folder}/${name}/${video_name_temp}
mkdir -p ${folder};
dest_file=${folder}/${video_name_temp}.mp4
ffmpeg -i $source_file -ss $start_s -t $size_s $dest_file
