#!/usr/bin/env python3
"""
Capture raw depth + RGB frames from an Intel RealSense D435i.

Install:  pip install pyrealsense2 numpy opencv-python

Usage:
    python capture_realsense.py                  # live preview, press 's' to save, 'q' to quit
    python capture_realsense.py --single         # grab one aligned frame and exit
    python capture_realsense.py --width 1280 --height 720 --fps 30 --out scenes_11_10_26

Saved per capture (in <out>/):
    rgb_XXXX.png        8-bit BGR color image
    depth_XXXX.png      16-bit filtered depth in sensor units (multiply by depth_scale -> meters)
    depth_XXXX.npy      same filtered depth as uint16 numpy array
    intrinsics.json     color intrinsics, depth scale (saved once)
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=str, default="scenes_11_10_26")
    p.add_argument("--single", action="store_true", help="capture one frame and exit")
    p.add_argument("--no-align", action="store_true",
                   help="do not align depth to color (keeps native depth geometry)")
    p.add_argument("--warmup", type=int, default=30,
                   help="frames to discard so auto-exposure settles")
    p.add_argument("--no-temporal-filter", action="store_true",
                   help="disable temporal smoothing when the camera or scene is moving")
    return p.parse_args()


def save_intrinsics(path, camera_profile, depth_scale):
    intr = camera_profile.as_video_stream_profile().get_intrinsics()
    data = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "distortion_model": str(intr.model),
        "distortion_coeffs": list(intr.coeffs),
        "depth_scale_m_per_unit": depth_scale,
        "K": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_frame(out_dir, idx, color, depth):
    cv2.imwrite(os.path.join(out_dir, f"rgb_{idx:04d}.png"), color)
    cv2.imwrite(os.path.join(out_dir, f"depth_{idx:04d}.png"), depth)  # 16-bit PNG
    np.save(os.path.join(out_dir, f"depth_{idx:04d}.npy"), depth)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # typically 0.001 m/unit for D435i
    print(f"Depth scale: {depth_scale} m/unit")

    # Align depth to the color frame so pixels correspond 1:1
    align = None if args.no_align else rs.align(rs.stream.color)
    to_disparity = rs.disparity_transform(True)
    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 2)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
    temporal_filter = None
    if not args.no_temporal_filter:
        temporal_filter = rs.temporal_filter()
        temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
    from_disparity = rs.disparity_transform(False)

    save_intrinsics(
        os.path.join(args.out, "intrinsics.json"),
        profile.get_stream(rs.stream.depth if args.no_align else rs.stream.color),
        depth_scale,
    )

    def process_frames(frames):
        if align is not None:
            frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return frames, None
        depth_frame = to_disparity.process(depth_frame)
        depth_frame = spatial_filter.process(depth_frame)
        if temporal_filter is not None:
            depth_frame = temporal_filter.process(depth_frame)
        depth_frame = from_disparity.process(depth_frame)
        return frames, depth_frame

    # Let auto-exposure and the temporal depth filter settle.
    for _ in range(args.warmup):
        process_frames(pipeline.wait_for_frames())

    temporal_status = "spatial + temporal" if temporal_filter is not None else "spatial"
    print(f"Depth filtering enabled: edge-preserving {temporal_status} filters.")

    idx = len([f for f in os.listdir(args.out) if f.startswith("rgb_")])

    try:
        while True:
            frames, depth_frame = process_frames(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())          # uint8, HxWx3 (BGR)
            depth = np.asanyarray(depth_frame.get_data())          # uint16, HxW (raw units)

            if args.single:
                save_frame(args.out, idx, color, depth)
                print(f"Saved frame {idx} to {args.out}/")
                break

            # Preview uses the same filtered depth that will be saved.
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )
            cv2.imshow("RealSense  [s] save  [q] quit", np.hstack((color, depth_vis)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_frame(args.out, idx, color, depth)
                print(f"Saved frame {idx}  (t={time.strftime('%H:%M:%S')})")
                idx += 1
            elif key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
