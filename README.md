# HumCentr-CLI: Human-Centric Computer Vision Tools for the Command Line

This is a collection of command line tools that were used in the processing of the [PosePile](https://github.com/isarandi/PosePile) dataset collection for 3D human pose estimation.

Specifically, the following human-centric computer vision tasks are tackled here:

* person detection (YOLOv4 - fork at https://github.com/isarandi/tensorflow-yolov4-tflite),
* person segmentation (DeepLabV3 - fork at https://github.com/isarandi/deeplabv3-tf2),
* 3D body pose estimation (MeTRAbs - repo at https://github.com/isarandi/metrabs).

The underlying models are in TensorFlow SavedModel format and are downloaded automatically on first use via TensorFlow Hub, and thus this is a standalone repository.

## Installation

```pip install git+https://github.com/isarandi/humcentr-cli.git```

## Examples

### Detection

On a directory of images:

```bash
python -m humcentr_cli.detect_people \
    --image-root="some/path/containing/images"\
    --out-path="some/other/path.pkl"
```

On a directory of videos:

```bash
python -m humcentr_cli.detect_people_video \
    --video-dir="some/path/containing/videos" \
    --output-dir="result/path" \
    --file-pattern='*.mp4' \
    --videos-per-task=2
```

### Segmentation

On a directory of images:

```bash 
SLURM_ARRAY_TASK_ID=0 python -m humcentr_cli.segment_people \ 
    --image-root="some/path/containing/images" \
    --out-dir="some/other/path" \
    --images-per-task=10000
```

### Pose Estimation

On a directory of images

```bash
python -m humcentr_cli.estimate_3d_pose \
    --model-path=https://bit.ly/metrabs_l \
    --image-root="path/to/images" \
    --file-pattern='**/*.jpg' \
    --out-path="path/to/outputs" \
    --camera-file="cameras.pkl" \
    --no-suppress-implausible-poses \
    --detector-threshold=0.1 \
    --no-average-aug \
    --horizontal-flip
```

On a directory of videos

```bash
SLURM_ARRAY_TASK_ID=0 python -m humcentr_cli.estimate_3d_pose_video
    --model-path=https://bit.ly/metrabs_l \
    --video-dir="path/to/videos" \
    --output-dir="path/to/outputs" \
    --file-pattern='*.mp4,*.avi' \
    --videos-per-task=5
```

See https://github.com/isarandi/PosePile for usage in practice.