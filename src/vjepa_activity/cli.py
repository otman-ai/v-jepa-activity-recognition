"""Video-file and webcam inference using V-JEPA2 + the pooled MLP."""

import argparse
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
from .model import load_pipeline
from .overlay import draw_action_overlay


def classify(frames, backbone, processor, classifier, classes, device):
    rgb = [cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB) for frame in frames]
    inputs = processor(videos=[np.asarray(rgb)], return_tensors="pt").to(device)
    with torch.inference_mode():
        features = backbone(**inputs).last_hidden_state.mean(dim=1)
        probabilities = torch.softmax(classifier(features), dim=1)[0]
        score, index = probabilities.max(dim=0)
    return classes[index.item()], score.item()*100


def process_video(source, output, pipeline, window, display):
    backbone, processor, classifier, classes, device = pipeline
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not total or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video metadata is invalid or the video is empty.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output}")
    recent = deque(maxlen=window)
    action, confidence = "Collecting frames", 0.0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            recent.append(frame.copy())
            if len(recent) == window and index % window == window-1:
                action, confidence = classify(list(recent), backbone, processor, classifier, classes, device)
            annotated = draw_action_overlay(frame, action, confidence, index, total, fps)
            writer.write(annotated)
            if display:
                cv2.imshow("V-JEPA2 Activity Recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            index += 1
    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()
    print(f"Saved annotated video: {output}")


def process_camera(camera, output, pipeline, window):
    backbone, processor, classifier, classes, device = pipeline
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {camera}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output}")
    recent = deque(maxlen=window)
    action, confidence, index = "Collecting frames", 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            recent.append(frame.copy())
            if len(recent) == window and index % window == window-1:
                action, confidence = classify(list(recent), backbone, processor, classifier, classes, device)
            annotated = draw_action_overlay(frame, action, confidence, index, index+1, fps)
            writer.write(annotated)
            cv2.imshow("V-JEPA2 Activity Recognition (press Q to stop)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            index += 1
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Saved camera recording: {output}")


def main():
    parser = argparse.ArgumentParser(description="Recognize UCF-101 activities from a video or camera.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", help="Path to input video")
    inputs.add_argument("--camera", nargs="?", const="0", help="Use camera (optional device index, default 0)")
    parser.add_argument("--output", help="Output annotated MP4")
    parser.add_argument("--checkpoint", default="checkpoints/classifier.pt")
    parser.add_argument("--backbone", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--window", type=int, default=32, help="Frames per prediction; use the training window of 32")
    parser.add_argument("--display", action="store_true", help="Show video-file inference in a window")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    pipeline = load_pipeline(args.checkpoint, args.backbone, args.device)
    if args.video:
        output = args.output or str(Path(args.video).with_name(Path(args.video).stem + "_annotated.mp4"))
        process_video(args.video, output, pipeline, args.window, args.display)
    else:
        camera = int(args.camera)
        process_camera(camera, args.output or "outputs/camera_annotated.mp4", pipeline, args.window)


if __name__ == "__main__":
    main()
