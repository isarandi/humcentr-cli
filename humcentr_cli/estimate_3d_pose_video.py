import argparse
import functools
import itertools
import os
import os.path as osp

import cameralib
import more_itertools
import numpy as np
import poseviz
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from simplepyutils import FLAGS

from humcentr_cli.util import improc, videoproc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='https://bit.ly/metrabs_l')
    parser.add_argument('--video-dir', type=str)
    parser.add_argument('--file-pattern', type=str, default='**/*.mp4')
    parser.add_argument('--videos-per-task', type=int, default=1)
    parser.add_argument('--every-nth-frame', type=int, default=1)
    parser.add_argument('--viz', action=spu_argparse.BoolAction)
    parser.add_argument('--viz-downscale', type=int, default=1)
    parser.add_argument('--high-quality-viz', action=spu_argparse.BoolAction)
    parser.add_argument('--audio', action=spu_argparse.BoolAction)
    parser.add_argument('--write-video', action=spu_argparse.BoolAction)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--camera-intrinsics-file', type=str)
    parser.add_argument('--camera-file', type=str)
    parser.add_argument('--fov', type=float, default=55)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=128)
    parser.add_argument('--max-detections', type=int, default=-1)
    parser.add_argument('--detector-threshold', type=float, default=0.2)
    parser.add_argument('--detector-nms-iou-threshold', type=float, default=0.7)
    parser.add_argument('--detector-flip-aug', action=spu_argparse.BoolAction)
    parser.add_argument('--antialias-factor', type=int, default=1)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--average-aug', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--skeleton', type=str, default='')
    parser.add_argument('--suppress-implausible-poses', action=spu_argparse.BoolAction)
    parser.add_argument('--enhance-green-room', action=spu_argparse.BoolAction)
    parser.add_argument('--gamma', type=float, default=1)
    spu.initialize(parser)

    globs = [spu.sorted_recursive_glob(f'{FLAGS.video_dir}/{p}')
             for p in FLAGS.file_pattern.split(',')]
    video_paths = [x for l in globs for x in l]
    try:
        i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
        i_start = i_task * FLAGS.videos_per_task
        video_paths = video_paths[i_start:i_start + FLAGS.videos_per_task]
    except KeyError:
        pass
    # random.shuffle(video_paths)
    relpaths = [osp.relpath(video_path, FLAGS.video_dir) for video_path in video_paths]
    camera_tracks = get_camera_calibrations(relpaths)

    output_paths = [
        f'{FLAGS.output_dir}/{osp.splitext(relpath)[0]}{FLAGS.suffix}.pkl'
        for relpath in relpaths]
    output_video_paths = [
        f'{FLAGS.output_dir}/{osp.splitext(relpath)[0]}{FLAGS.suffix}.mp4'
        for relpath in relpaths]

    if (all(spu.is_pickle_readable(p) for p in output_paths) and
            (not FLAGS.write_video or all(osp.exists(p) for p in output_video_paths))):
        print('All done')
        return

    video_slice = slice(0, None, FLAGS.every_nth_frame)

    model = tfhub.load(FLAGS.model_path)
    predict_fn = functools.partial(
        model.detect_poses_batched, default_fov_degrees=FLAGS.fov,
        detector_threshold=FLAGS.detector_threshold, num_aug=FLAGS.num_aug,
        detector_nms_iou_threshold=FLAGS.detector_nms_iou_threshold,
        max_detections=FLAGS.max_detections, internal_batch_size=FLAGS.internal_batch_size,
        skeleton=FLAGS.skeleton, antialias_factor=FLAGS.antialias_factor,
        average_aug=FLAGS.average_aug, suppress_implausible_poses=FLAGS.suppress_implausible_poses,
        detector_flip_aug=FLAGS.detector_flip_aug)

    joint_names = model.per_skeleton_joint_names[FLAGS.skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[FLAGS.skeleton].numpy()

    first_camera, camera_tracks = spu.nested_spy_first(camera_tracks, levels=2)
    up = first_camera.world_up if first_camera is not None else np.array((0, -1, 0), np.float32)
    viz = poseviz.PoseViz(
        joint_names, joint_edges, world_up=up, high_quality=FLAGS.high_quality_viz,
        downscale=FLAGS.viz_downscale, show_ground_plane=True, show_field_of_view=True,
        resolution=(1920, 1080) if FLAGS.high_quality_viz else (1280, 720),
        camera_view_padding=0.2, show_camera_wireframe=True,
        draw_detections=True) if FLAGS.viz else None

    frame_preproc_fn = (
        enhance_green_room if FLAGS.enhance_green_room else (
            adjust_gamma if FLAGS.gamma != 1 else None))

    for video_path, output_path, output_video_path, camera_track in zip(
            video_paths, output_paths, output_video_paths, camera_tracks):
        if (spu.is_pickle_readable(output_path) and
                (not FLAGS.write_video or osp.exists(output_video_path))):
            continue
        print(video_path)

        if FLAGS.viz:
            viz.reinit_camera_view()
            if FLAGS.write_video:
                viz.new_sequence_output(output_video_path, fps=videoproc.get_fps(video_path))

        frame_batches, frame_batches_cpu = tfinp.video_file(
            video_path, batch_size=FLAGS.batch_size, video_slice=video_slice, tee_cpu=FLAGS.viz,
            frame_preproc_fn=frame_preproc_fn)
        n_frames = videoproc.num_frames(video_path)

        camera_batches = more_itertools.chunked(camera_track, FLAGS.batch_size)
        batch_preds = []
        progbar = spu.progressbar(total=n_frames, unit=' frames')
        for frame_batch, frame_batch_cpu, camera_batch in zip(
                frame_batches, frame_batches_cpu, camera_batches):
            default_cam = cameralib.Camera.from_fov(FLAGS.fov, frame_batch[0].shape)
            default_cam.world_up = up

            batch_size = frame_batch.shape[0]
            camera_batch = [c if c is not None else default_cam for c in camera_batch][:batch_size]
            intrinsics = np.array([c.intrinsic_matrix for c in camera_batch], np.float32)
            extrinsics = np.array([c.get_extrinsic_matrix() for c in camera_batch], np.float32)
            distortion_coeffs = np.array(
                [c.get_distortion_coeffs() for c in camera_batch], np.float32)
            pred = predict_fn(
                frame_batch, intrinsic_matrix=intrinsics, extrinsic_matrix=extrinsics,
                distortion_coeffs=distortion_coeffs, world_up_vector=up)

            pred = tf.nest.map_structure(lambda x: x.numpy(), pred)

            if FLAGS.viz:
                for frame, boxes, poses, camera in zip(
                        frame_batch_cpu, pred['boxes'], pred['poses3d'], camera_batch):
                    viz.update(frame, boxes, poses, camera)

            progbar.update(frame_batch.shape[0])
            batch_preds.append(pred)

        if FLAGS.output_dir:
            boxes = [b for p in batch_preds for b in p['boxes']]
            poses2d = [b for p in batch_preds for b in p['poses2d']]
            poses3d = [b for p in batch_preds for b in p['poses3d']]
            spu.dump_pickle(dict(boxes=boxes, poses2d=poses2d, poses3d=poses3d), output_path)

        if FLAGS.write_video and FLAGS.audio:
            viz.finalize_sequence_output()
            videoproc.video_audio_mux(
                vidpath_audiosource=video_path, vidpath_imagesource=output_video_path,
                out_video_path=spu.replace_extension(output_video_path, '_audio.mp4'))

    if FLAGS.viz:
        viz.close()


def get_camera_calibrations(relpaths):
    if FLAGS.camera_intrinsics_file:
        intrinsics = spu.load_pickle(FLAGS.camera_intrinsics_file)
        if isinstance(intrinsics, dict):
            intrinsics = [intrinsics[relpath] for relpath in relpaths]
        else:
            intrinsics = itertools.repeat(intrinsics)
        return (itertools.repeat(cameralib.Camera(intrinsic_matrix=intr, world_up=(0, -1, 0)))
                for intr in intrinsics)

    if FLAGS.camera_file:
        camera = spu.load_pickle(FLAGS.camera_file)
        if isinstance(camera, dict):
            return [get_camera_track(camera[relpath]) for relpath in relpaths]
        else:
            return itertools.repeat(get_camera_track(camera))

    return itertools.repeat(itertools.repeat(None))


def get_camera_track(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, cameralib.Camera):
        return itertools.repeat(x)
    else:
        raise Exception('Invalid camera file format')


def enhance_green_room(frame):
    frame = improc.adjust_gamma(frame, 0.67, inplace=True)
    frame = improc.white_balance(frame, 110, 145)
    return frame


def adjust_gamma(frame):
    return improc.adjust_gamma(frame, FLAGS.gamma, inplace=True)


if __name__ == '__main__':
    main()
