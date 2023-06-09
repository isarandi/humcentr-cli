"""Runs binary semantic segmentation for the person category on image files and saves the
results."""
import argparse
import glob
import math
import os

import more_itertools
import rlemasklib
import simplepyutils as spu
import tensorflow as tf
import tensorflow_hub as tfhub
from simplepyutils import FLAGS, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=str)
    parser.add_argument('--model-path', type=str,
                        default='https://github.com/isarandi/deeplabv3-tf2/releases/download/v0.1.0'
                                '/deeplabv3_pascal_person.tar.gz')
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--images-per-task', type=int, default=10000)
    parser.add_argument('--file-pattern', type=str, default='**/*.jpg')
    parser.add_argument('--mask-threshold', type=float, default=0.2)
    spu.initialize(parser)

    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    result_path = f'{FLAGS.out_dir}/masks_{i_task:06d}.pkl'

    if os.path.exists(result_path):
        return

    globs = [glob.glob(f'{FLAGS.image_root}/{p}', recursive=True)
             for p in FLAGS.file_pattern.split(',')]
    impaths = sorted([x for l in globs for x in l])
    n_jobs = math.ceil(len(impaths) / FLAGS.images_per_task)
    logger.info(f'{len(impaths)} images found, {n_jobs} jobs needed.')
    impaths = impaths[i_task * FLAGS.images_per_task:(i_task + 1) * FLAGS.images_per_task]
    frames = (
        tf.data.Dataset.from_tensor_slices(impaths).
        map(load_jpeg).
        prefetch(2 * FLAGS.batch_size))

    model = tfhub.load(FLAGS.model_path)
    impath_batches = more_itertools.chunked(spu.progressbar(impaths), FLAGS.batch_size)
    frame_batches = more_itertools.chunked(frames, FLAGS.batch_size)
    results = {}
    for impaths_batch, frames_batch in zip(impath_batches, frame_batches):
        masks_batch = segment_batch(model, frames_batch)
        for impath, mask in zip(impaths_batch, masks_batch):
            relpath = os.path.relpath(impath, FLAGS.image_root)
            results[relpath] = rlemasklib.encode(mask.numpy())

    spu.dump_pickle(results, result_path)


def segment_batch(model, images):
    """Segment a "batch" of images that may each have different sizes by padded resizing,
    predicting, and converting the resulting mask back to the original resolution by unpadding and
    resizing"""
    inp = tf.stack([resize_with_pad(im, (513, 513)) for im in images])
    outp = model(inp)[..., tf.newaxis]
    masks = [resize_with_unpad(mask, im.shape[:2]) for mask, im in zip(outp, images)]
    masks = [tf.cast(tf.squeeze(mask, -1) > FLAGS.mask_threshold, tf.uint8) for mask in masks]
    return masks


def load_jpeg(path, ratio=1):
    return tf.image.decode_jpeg(
        tf.io.read_file(path), fancy_upscaling=False, dct_method='INTEGER_FAST', ratio=ratio)


def resize_with_pad(image, target_shape):
    if image.ndim == 3:
        return tf.squeeze(resize_with_pad(image[tf.newaxis], target_shape), 0)
    factor, target_shape_part, rest_shape = resized_size_and_rest(
        tf.shape(image)[1:3], target_shape)
    if factor > 1:
        image = tf.cast(tf.image.resize(
            image, target_shape_part, method=tf.image.ResizeMethod.BILINEAR), image.dtype)
    else:
        image = tf.cast(tf.image.resize(
            image, target_shape_part, method=tf.image.ResizeMethod.AREA), image.dtype)

    return tf.pad(image, [(0, 0), (rest_shape[0], 0), (rest_shape[1], 0), (0, 0)])


def resize_with_unpad(image, orig_shape):
    if image.ndim == 3:
        return tf.squeeze(resize_with_unpad(image[tf.newaxis], orig_shape), 0)
    factor, _, rest_shape = resized_size_and_rest(orig_shape, tf.shape(image)[1:3])
    image = image[:, rest_shape[0]:, rest_shape[1]:]
    if factor < 1:
        image = tf.cast(
            tf.image.resize(
                image, orig_shape, method=tf.image.ResizeMethod.BILINEAR), image.dtype)
    else:
        image = tf.cast(
            tf.image.resize(
                image, orig_shape, method=tf.image.ResizeMethod.AREA), image.dtype)
    return image


def resized_size_and_rest(input_shape, target_shape):
    target_shape_float = tf.cast(target_shape, tf.float32)
    input_shape_float = tf.cast(input_shape, tf.float32)
    factor = tf.reduce_min(target_shape_float / input_shape_float)
    target_shape_part = tf.cast(factor * input_shape_float, tf.int32)
    rest_shape = target_shape - target_shape_part
    return factor, target_shape_part, rest_shape


if __name__ == '__main__':
    main()
