import argparse
import functools
import itertools
import os

import cameralib
import more_itertools
import numpy as np
import poseviz
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from simplepyutils.argparse import FLAGS

from humcentr_cli.detect_people import get_task_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=str, required=True)
    parser.add_argument('--ignore-paths-file', type=str)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--file-pattern', type=str, default='**/*.jpg')
    parser.add_argument('--images-per-task', type=int, default=10000)
    parser.add_argument('--image-type', type=str, default='jpg')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--camera-intrinsics-file', type=str)
    parser.add_argument('--camera-file', type=str)
    parser.add_argument('--internal-batch-size', type=int, default=128)
    parser.add_argument('--max-detections', type=int, default=-1)
    parser.add_argument('--detector-threshold', type=float, default=0.2)
    parser.add_argument('--detector-nms-iou-threshold', type=float, default=0.7)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--fov', type=float, default=55)
    parser.add_argument('--average-aug', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--skeleton', type=str, default='')
    parser.add_argument('--suppress-implausible-poses', action=spu_argparse.BoolAction)
    parser.add_argument('--viz', action=spu_argparse.BoolAction)
    parser.add_argument('--horizontal-flip', action=spu_argparse.BoolAction)
    parser.add_argument('--high-quality-viz', action=spu_argparse.BoolAction)
    parser.add_argument('--antialias-factor', type=int, default=2)
    spu_argparse.initialize(parser)

    image_relpaths = get_image_relpaths()
    image_relpaths, out_path = get_task_chunk(image_relpaths)
    if os.path.exists(out_path) or not image_relpaths:
        return

    model = tfhub.load(FLAGS.model_path)
    predict_fn = functools.partial(
        model.detect_poses_batched, default_fov_degrees=FLAGS.fov,
        detector_threshold=FLAGS.detector_threshold, num_aug=FLAGS.num_aug,
        detector_nms_iou_threshold=FLAGS.detector_nms_iou_threshold,
        max_detections=FLAGS.max_detections, internal_batch_size=FLAGS.internal_batch_size,
        skeleton=FLAGS.skeleton, antialias_factor=FLAGS.antialias_factor,
        average_aug=FLAGS.average_aug, suppress_implausible_poses=FLAGS.suppress_implausible_poses)

    relpath_to_cam = dict(zip(image_relpaths, get_camera_calibrations(image_relpaths)))
    camera_to_relpaths = spu.groupby(image_relpaths, relpath_to_cam.get)
    cameras = list(camera_to_relpaths.keys())

    frame_preproc_fn = (lambda x: x[:, ::-1]) if FLAGS.horizontal_flip else None

    joint_names = model.per_skeleton_joint_names[FLAGS.skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[FLAGS.skeleton].numpy()
    world_up = cameras[0].world_up if cameras[0] is not None else (0, -1, 0)
    viz = poseviz.PoseViz(
        joint_names, joint_edges, world_up=world_up, paused=True) if FLAGS.viz else None

    results = {}
    for camera, relpaths_of_seq in camera_to_relpaths.items():
        predict_fn_cam = (
            functools.partial(
                predict_fn, intrinsic_matrix=camera.intrinsic_matrix[np.newaxis],
                extrinsic_matrix=camera.get_extrinsic_matrix()[np.newaxis],
                distortion_coeffs=camera.get_distortion_coeffs()[np.newaxis],
                world_up_vector=camera.world_up)
            if camera is not None else predict_fn)

        paths_of_seq = [f'{FLAGS.image_root}/{p}' for p in relpaths_of_seq]
        frame_batches, frame_batches_cpu = tfinp.image_files(
            paths_of_seq, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz,
            frame_preproc_fn=frame_preproc_fn, internal_queue_size=5, prefetch_gpu=2)

        relpath_batches = more_itertools.chunked(
            spu.progressbar(relpaths_of_seq), FLAGS.batch_size)

        for relpath_batch, image_batch in zip(relpath_batches, frame_batches):
            pred = predict_fn_cam(image_batch)
            pred = tf.nest.map_structure(lambda x: x.numpy(), pred)
            if FLAGS.viz:
                for frame, boxes, poses in zip(
                        image_batch.numpy(), pred['boxes'], pred['poses3d']):
                    viz_cam = (
                        camera if camera is not None
                        else cameralib.Camera.from_fov(55, frame.shape))
                    viz.update(frame, boxes, poses, viz_cam)

            for relpath, boxes, poses3d in zip(relpath_batch, pred['boxes'], pred['poses3d']):
                results[relpath] = dict(boxes=boxes, poses3d=poses3d)

    spu.dump_pickle(results, out_path)

    if FLAGS.viz:
        viz.close()


def get_image_relpaths():
    globs = [spu.sorted_recursive_glob(f'{FLAGS.image_root}/{p}')
             for p in FLAGS.file_pattern.split(',')]
    image_paths = sorted([x for l in globs for x in l])
    image_relpaths = [os.path.relpath(p, FLAGS.image_root) for p in image_paths]
    ignore_relpaths = set(
        spu.read_file(FLAGS.ignore_paths_file).splitlines() if FLAGS.ignore_paths_file else [])
    image_relpaths = sorted([p for p in image_relpaths if p not in ignore_relpaths])
    return image_relpaths


def get_camera_calibrations(relpaths):
    if FLAGS.camera_intrinsics_file:
        intrinsics = spu.load_pickle(FLAGS.camera_intrinsics_file)
        if isinstance(intrinsics, dict):
            intrinsics = [intrinsics[relpath] for relpath in relpaths]
        else:
            intrinsics = itertools.repeat(intrinsics)
        return (cameralib.Camera(intrinsic_matrix=intr, world_up=(0, -1, 0)) for intr in intrinsics)

    if FLAGS.camera_file:
        camera = spu.load_pickle(FLAGS.camera_file)
        if isinstance(camera, dict):
            return [camera[relpath] for relpath in relpaths]
        else:
            return itertools.repeat(camera)

    return itertools.repeat(None)


if __name__ == '__main__':
    main()
