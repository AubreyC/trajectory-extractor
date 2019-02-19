# Object detection with Mask-RCNN

## Installation and Requirements

Update & download submodules:

    git submodule update --init --update

 The following instructions are an adapted version of the installation steps from Mask-RCNN: https://github.com/matterport/Mask_RCNN

**Step 1**: Install `pycocotools`

    cd Mask_RCNN
    git clone https://github.com/waleedka/coco
    cd coco/PythonAPI
    make
    python setup.py install

**Step 2**: Download the COCO trained weights

    cd Mask_RCNN
    wget 'https://cloud.mines-paristech.fr/index.php/s/YSxeVAkO2cElAE8/download'

*Note:* This is saved version of the COCO trained weights.

## Run

Run `run_detections_csv.py` to run the `mask-rcnn` on all the image from the image folder.

Please fill `DETECTOR_MRCNN_CFG.ini` with:
- Directory with the raw images
- Output directory to save the csv and annotated images
- Options