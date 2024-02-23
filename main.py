import os
import cv2
import numpy as np
import pandas as pd
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO

CURRENT_DIRECTORY = os.getcwd()
SOURCE_VIDEO_PATH = ''  # add your video path
TARGET_VIDEO_PATH = ''  # add your video path
MODEL = os.path.join(CURRENT_DIRECTORY, 'weight/yolov8x.pt')

model = YOLO(MODEL)
model.fuse()

LINE_START = sv.Point(
    # add your start line point
)
LINE_END = sv.Point(
    # add your end line point
)

polygon = np.array([
    # add your polygon
])

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps,
    track_thresh=0.30,
    match_thresh=0.8,
    track_buffer=video_info.fps
)
line_zone = sv.LineZone(
    start=LINE_START,
    end=LINE_END
)
bounding_box_annotator = sv.BoundingBoxAnnotator(
    thickness=1,
    color_lookup=sv.ColorLookup.TRACK
)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    color_lookup=sv.ColorLookup.TRACK
)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5
)
zone = sv.PolygonZone(
    polygon=polygon,
    frame_resolution_wh=video_info.resolution_wh)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
vid_writer = cv2.VideoWriter(
    TARGET_VIDEO_PATH,
    fourcc,
    video_info.fps,
    (video_info.width, video_info.height))
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

for _ in tqdm(range(video_info.total_frames), desc="Rendering videos with Bounding Box: "):
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        classes=[2, 5, 7],
        verbose=False,
        device=''  # set device as yours
    )[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = detections[zone.trigger(detections)]
    detections = byte_track.update_with_detections(detections)

    labels = [
        f"{confidence:0.2f}"
        for confidence
        in detections.confidence
    ]

    line_zone.trigger(detections)

    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    annotated_frame = line_zone_annotator.annotate(
        annotated_frame,
        line_counter=line_zone
    )

    vid_writer.write(annotated_frame)

cap.release()
vid_writer.release()
