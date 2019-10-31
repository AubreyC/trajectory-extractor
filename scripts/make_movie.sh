source_folder="test_data/brest_20190609_130424_327_334/output/visualizer"
name="brest_20190609_130424_327_334"

img_folder="img_hdmap"
cd ${source_folder}/${img_folder}
ffmpeg  -framerate 10 -pattern_type glob -i '*.png' video.mp4
cd ${source_folder}
mv ${source_folder}/${img_folder}/video.mp4 ${source_folder}/${name}_hdmap.mp4

img_folder="img_annoted"
cd ${source_folder}/${img_folder}
ffmpeg  -framerate 10 -pattern_type glob -i '*.png' video.mp4
cd ${source_folder}
mv ${source_folder}/${img_folder}/video.mp4 ${source_folder}/${name}_annotated.mp4