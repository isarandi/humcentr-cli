"""Runs person bounding box detection in image files and saves the results."""
import argparse
import os

import more_itertools
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import tensorflow as tf
import tensorflow_hub as tfhub
from simplepyutils.argparse import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=str, required=True)
    parser.add_argument('--ignore-paths-file', type=str)
    parser.add_argument('--image-paths-file', type=str)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--model-path', type=str,
                        default='https://github.com/isarandi/tensorflow-yolov4-tflite/releases'
                                '/download/v0.1.0/yolov4_416.tar.gz')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--nms-iou-threshold', type=float, default=0.7)
    parser.add_argument('--file-pattern', type=str, default='**/*.jpg')
    parser.add_argument('--images-per-task', type=int, default=10000)
    parser.add_argument('--image-type', type=str, default='jpg')
    parser.add_argument('--rot', type=int, default=0)
    spu_argparse.initialize(parser)

    if not FLAGS.image_paths_file:
        glob_lists = [spu.sorted_recursive_glob(f'{FLAGS.image_root}/{p}')
                      for p in FLAGS.file_pattern.split(',')]
        image_paths = sorted([x for glob_list in glob_lists for x in glob_list])
        image_relpaths = [os.path.relpath(p, FLAGS.image_root) for p in image_paths]
    else:
        image_relpaths = spu.read_lines(FLAGS.image_paths_file)

    ignore_relpaths = set(
        spu.read_lines(FLAGS.ignore_paths_file) if FLAGS.ignore_paths_file else [])
    image_relpaths = sorted([p for p in image_relpaths
                             if p not in ignore_relpaths and 'stitched' not in p])
    image_relpaths, out_path = get_task_chunk(image_relpaths)
    if os.path.exists(out_path):
        return
    if not image_relpaths:
        return

    ds = (tf.data.Dataset.from_tensor_slices(image_relpaths).
          map(load_image).
          batch(FLAGS.batch_size).
          apply(tf.data.experimental.prefetch_to_device('GPU:0', 2)))

    detector = tfhub.load(FLAGS.model_path)
    results = {}
    relpath_batches = more_itertools.chunked(spu.progressbar(image_relpaths), FLAGS.batch_size)
    for relpath_batch, image_batch in zip(relpath_batches, ds):
        detections_batch = detector.predict_multi_image(
            image_batch, FLAGS.threshold, FLAGS.nms_iou_threshold)
        if FLAGS.rot:
            rot_detections_back(detections_batch, image_batch.shape[1:3])
        results.update(dict(zip(relpath_batch, detections_batch.numpy())))

    spu.dump_pickle(results, out_path)


def rot_detections_back(det, rotated_imshape):
    H, W = rotated_imshape[:2]
    x, y, w, h = det[..., 0], det[..., 1], det[..., 2], det[..., 3]
    if FLAGS.rot == 90:
        return tf.stack([H - (y + h), x, h, w], axis=2)
    elif FLAGS.rot == 180:
        return tf.stack([W - (x + w), H - (y + h), w, h], axis=2)
    elif FLAGS.rot == 270:
        return tf.stack([y, W - (x + w), h, w], axis=2)
    else:
        raise ValueError(f'Invalid rotation {FLAGS.rot}, must be one of 0, 90, 180, 270')


def get_task_chunk(image_paths):
    if 'SLURM_ARRAY_TASK_ID' not in os.environ:
        return image_paths, FLAGS.out_path
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    noext, ext = os.path.splitext(FLAGS.out_path)
    out_path = f'{noext}_{i_task:06d}{ext}'
    task_impaths = image_paths[i_task * FLAGS.images_per_task:(i_task + 1) * FLAGS.images_per_task]
    return task_impaths, out_path


def load_image(relpath):
    im = load_jpg(relpath) if FLAGS.image_type == 'jpg' else load_png(relpath)
    if FLAGS.rot:
        im = tf.image.rot90(im, k=FLAGS.rot // 90)
    return im


def load_jpg(relpath):
    return tf.image.decode_jpeg(tf.io.read_file(FLAGS.image_root + '/' + relpath))


def load_png(relpath):
    return tf.image.decode_png(tf.io.read_file(FLAGS.image_root + '/' + relpath))[..., :3]


if __name__ == '__main__':
    main()
