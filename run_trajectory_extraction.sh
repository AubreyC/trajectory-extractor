 #!/bin/bash

# SET THE PATH AND CONFIG
SOURCE_FOLDER="test_dataset/brest_20190609_130424_327_334"
VIDEO_NAME="brest_20190609_130424_327_334.mp4"
NAME="brest_20190609_130424_327_334"
DELTA_MS=100
LOCATION_NAME="brest"
DATE="20190609"
START_TIME="1310"

CAMERA_STREET="brest_area1_street_cfg.yml"
CAMERA_SAT="brest_area1_sat_cfg.yml"
CAMERA_SAT_IMG="brest_area1_sat.png"
HD_MAP="brest_area1_street_hd_map.csv"

# SET DETECTION AD IGNORE ZONES
DET_ZONE_IM_VEHICLES="brest_area1_detection_zone_im.yml"
DET_ZONE_FNED_VEHICLES="brest_area1_detection_zone.yml"
IGNORE_AREA_VEHICLES=""

DET_ZONE_IM_PEDESTRIANS="brest_area1_detection_zone_im.yml"
DET_ZONE_FNED_PEDESTRIANS="brest_area1_detection_zone.yml"
IGNORE_AREA_PEDESTRIANS="brest_area1_ignoreareas.csv"

# SET CROP VALUES FOR DETECTION HERE IF NEEDED
CROP_X1=180
CROP_Y1=120
CROP_X2=1250
CROP_Y2=720

IMG_DIR=${SOURCE_FOLDER}"/img"
OUTPUT_DIR=${SOURCE_FOLDER}"/output"

DET_DIR=${OUTPUT_DIR}"/det/csv"

MODE_VEHICLES="vehicles"
DYNAMIC_MODEL_VEHICLES="BM2"
LABEL_REPLACE_VEHICLES="car"
OUTPUT_VEHICLES_DIR=${OUTPUT_DIR}/${MODE_VEHICLES}
DET_ASSO_VEHICLES_DIR=${OUTPUT_VEHICLES_DIR}"/det_association/csv"
TRAJ_VEHICLES_DIR=${OUTPUT_VEHICLES_DIR}"/traj/csv"
TRACK_MERGE_VEHICLES=${OUTPUT_VEHICLES_DIR}"/det_association/"${NAME}"_tracks_merge.csv"
TRAJ_VEHICLES=${TRAJ_VEHICLES_DIR}/${NAME}"_traj.csv"
TRAJ_INSPECT_VEHICLES_DIR=${OUTPUT_VEHICLES_DIR}"/traj_inspect/csv"
TRAJ_INSPECT_VEHICLES=${TRAJ_INSPECT_VEHICLES_DIR}/${NAME}"_traj.csv"

MODE_PEDESTRIANS="pedestrians"
DYNAMIC_MODEL_PEDESTRIANS="CV"
LABEL_REPLACE_PEDESTRIANS="person"
OUTPUT_PEDESTRIANS_DIR=${OUTPUT_DIR}/${MODE_PEDESTRIANS}
DET_ASSO_PEDESTRIANS_DIR=${OUTPUT_PEDESTRIANS_DIR}"/det_association/csv"
TRAJ_PEDESTRIANS_DIR=${OUTPUT_PEDESTRIANS_DIR}"/traj/csv"
TRACK_MERGE_PEDESTRIANS=${OUTPUT_PEDESTRIANS_DIR}"/det_association/"${NAME}"_tracks_merge.csv"
TRAJ_PEDESTRIANS=${TRAJ_PEDESTRIANS_DIR}/${NAME}"_traj.csv"
TRAJ_INSPECT_PEDESTRIANS_DIR=${OUTPUT_PEDESTRIANS_DIR}"/traj_inspect/csv"
TRAJ_INSPECT_PEDESTRIANS=${TRAJ_INSPECT_PEDESTRIANS_DIR}/${NAME}"_traj.csv"

##################################################################
# EXTRACTING FRAMES FROM VIDEO
##################################################################

VIDEO_PATH=${SOURCE_FOLDER}/${VIDEO_NAME}
python traj_ext/object_det/run_saveimages.py ${VIDEO_PATH} --skip 3

####################################################################
# OBJECT DETECTION
####################################################################

python traj_ext/object_det/mask_rcnn/run_detections_csv.py\
        -image_dir ${IMG_DIR}\
        -output_dir ${OUTPUT_DIR}\
        -crop_x1y1x2y2 ${CROP_X1} ${CROP_Y1} ${CROP_X2} ${CROP_Y2}\
        -no_save_images

####################################################################
# VEHICLES
####################################################################

# Det association
python traj_ext/det_association/run_det_association.py\
            -image_dir ${IMG_DIR}\
            -output_dir ${OUTPUT_VEHICLES_DIR}\
            -det_dir ${DET_DIR}\
            -ignore_detection_area ${SOURCE_FOLDER}/${IGNORE_AREA_VEHICLES}\
            -det_zone_im ${SOURCE_FOLDER}/${DET_ZONE_IM_VEHICLES}\
            -mode ${MODE_VEHICLES}\
            -no_save_images

# Process
python traj_ext/postprocess_track/run_postprocess.py\
            -image_dir ${IMG_DIR}\
            -output_dir ${OUTPUT_VEHICLES_DIR}\
            -det_dir ${DET_DIR}\
            -det_asso_dir ${DET_ASSO_VEHICLES_DIR}\
            -track_merge ${TRACK_MERGE_VEHICLES}\
            -camera_street ${SOURCE_FOLDER}/${CAMERA_STREET}\
            -camera_sat  ${SOURCE_FOLDER}/${CAMERA_SAT}\
            -camera_sat_img ${SOURCE_FOLDER}/${CAMERA_SAT_IMG}\
            -det_zone_fned ${SOURCE_FOLDER}/${DET_ZONE_FNED_VEHICLES}\
            -delta_ms ${DELTA_MS}\
            -location_name ${LOCATION_NAME}\
            -dynamic_model ${DYNAMIC_MODEL_VEHICLES}\
            -date ${DATE}\
            -start_time ${START_TIME}\
            -no_save_images

python traj_ext/visualization/run_inspect_traj.py\
            -traj ${TRAJ_VEHICLES}\
            -image_dir ${IMG_DIR}\
            -det_dir ${DET_DIR}\
            -det_asso_dir ${DET_ASSO_VEHICLES_DIR}\
            -track_merge ${TRACK_MERGE_VEHICLES}\
            -camera_street ${SOURCE_FOLDER}/${CAMERA_STREET}\
            -camera_sat  ${SOURCE_FOLDER}/${CAMERA_SAT}\
            -camera_sat_img ${SOURCE_FOLDER}/${CAMERA_SAT_IMG}\
            -det_zone_fned ${SOURCE_FOLDER}/${DET_ZONE_FNED_VEHICLES}\
            -label_replace ${LABEL_REPLACE_VEHICLES}\
            -output_dir ${OUTPUT_VEHICLES_DIR}\
            -hd_map ${SOURCE_FOLDER}/${HD_MAP}\
            -delta_ms ${DELTA_MS}\
            -location_name ${LOCATION_NAME}\
            -date ${DATE}\
            -start_time ${START_TIME}\
            -export


###################################################################
# PEDESTRIAN
###################################################################

# Det association
python traj_ext/det_association/run_det_association.py\
            -image_dir ${IMG_DIR}\
            -output_dir ${OUTPUT_PEDESTRIANS_DIR}\
            -det_dir ${DET_DIR}\
            -ignore_detection_area ${SOURCE_FOLDER}/${IGNORE_AREA_PEDESTRIANS}\
            -det_zone_im ${SOURCE_FOLDER}/${DET_ZONE_IM_PEDESTRIANS}\
            -mode ${MODE_PEDESTRIANS}\
            -no_save_images

# Process
python traj_ext/postprocess_track/run_postprocess.py\
            -image_dir ${IMG_DIR}\
            -output_dir ${OUTPUT_PEDESTRIANS_DIR}\
            -det_dir ${DET_DIR}\
            -det_asso_dir ${DET_ASSO_PEDESTRIANS_DIR}\
            -track_merge ${TRACK_MERGE_PEDESTRIANS}\
            -camera_street ${SOURCE_FOLDER}/${CAMERA_STREET}\
            -camera_sat  ${SOURCE_FOLDER}/${CAMERA_SAT}\
            -camera_sat_img ${SOURCE_FOLDER}/${CAMERA_SAT_IMG}\
            -det_zone_fned ${SOURCE_FOLDER}/${DET_ZONE_FNED_PEDESTRIANS}\
            -delta_ms ${DELTA_MS}\
            -location_name ${LOCATION_NAME}\
            -dynamic_model ${DYNAMIC_MODEL_PEDESTRIANS}\
            -date ${DATE}\
            -start_time ${START_TIME}\
            -no_save_images

python traj_ext/visualization/run_inspect_traj.py\
            -traj ${TRAJ_PEDESTRIANS}\
            -image_dir ${IMG_DIR}\
            -det_dir ${DET_DIR}\
            -det_asso_dir ${DET_ASSO_PEDESTRIANS_DIR}\
            -track_merge ${TRACK_MERGE_PEDESTRIANS}\
            -camera_street ${SOURCE_FOLDER}/${CAMERA_STREET}\
            -camera_sat  ${SOURCE_FOLDER}/${CAMERA_SAT}\
            -camera_sat_img ${SOURCE_FOLDER}/${CAMERA_SAT_IMG}\
            -det_zone_fned ${SOURCE_FOLDER}/${DET_ZONE_FNED_PEDESTRIANS}\
            -label_replace ${LABEL_REPLACE_PEDESTRIANS}\
            -output_dir ${OUTPUT_PEDESTRIANS_DIR}\
            -hd_map ${SOURCE_FOLDER}/${HD_MAP}\
            -delta_ms ${DELTA_MS}\
            -location_name ${LOCATION_NAME}\
            -date ${DATE}\
            -start_time ${START_TIME}\
            -export


###################################################################
# VISUALIZATION
###################################################################

python traj_ext/visualization/run_visualizer.py\
            -traj ${TRAJ_INSPECT_VEHICLES}\
            -traj_person ${TRAJ_INSPECT_PEDESTRIANS}\
            -image_dir ${IMG_DIR}\
            -camera_street ${SOURCE_FOLDER}/${CAMERA_STREET}\
            -camera_sat  ${SOURCE_FOLDER}/${CAMERA_SAT}\
            -camera_sat_img ${SOURCE_FOLDER}/${CAMERA_SAT_IMG}\
            -det_zone_fned ${SOURCE_FOLDER}/${DET_ZONE_FNED_VEHICLES}\
            -hd_map ${SOURCE_FOLDER}/${HD_MAP}\
            -output_dir ${OUTPUT_DIR}\