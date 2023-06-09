"""Runs person bounding box detection in video files and saves the results."""

import argparse
import os

import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from simplepyutils.argparse import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', type=str,
        default='https://github.com/isarandi/tensorflow-yolov4-tflite/releases'
                '/download/v0.1.0/yolov4_416.tar.gz')
    parser.add_argument('--video-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--nms-iou-threshold', type=float, default=0.7)
    parser.add_argument('--file-pattern', type=str, default='**/*.mp4')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--videos-per-task', type=int, default=1)
    parser.add_argument('--every-nth-frame', type=int, default=1)
    spu_argparse.initialize(parser)

    globs = [spu.sorted_recursive_glob(f'{FLAGS.video_dir}/{p}')
             for p in FLAGS.file_pattern.split(',')]
    video_paths = sorted([x for l in globs for x in l])
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    video_paths = video_paths[i_task * FLAGS.videos_per_task:(i_task + 1) * FLAGS.videos_per_task]
    relpaths = [os.path.relpath(video_path, FLAGS.video_dir) for video_path in video_paths]
    output_paths = [f'{FLAGS.output_dir}/{os.path.splitext(relpath)[0]}.pkl'
                    for relpath in relpaths]
    if all(os.path.exists(p) for p in output_paths):
        return

    video_slice = slice(0, None, FLAGS.every_nth_frame)
    model = tfhub.load(FLAGS.model_path)

    for video_path, output_path in zip(video_paths, output_paths):
        print(video_path)
        if os.path.exists(output_path):
            continue

        preds = []
        for frame_batch in spu.progressbar(tfinp.video_file(
                video_path, batch_size=FLAGS.batch_size, video_slice=video_slice)[0]):
            pred = model.predict_multi_image(frame_batch, FLAGS.threshold, FLAGS.nms_iou_threshold)
            preds.extend(pred.numpy())
        spu.dump_pickle(preds, output_path)


if __name__ == '__main__':
    main()
