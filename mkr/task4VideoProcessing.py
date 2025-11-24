from ultralytics import YOLO
import cv2
from pathlib import Path

MODEL_PATH = "/kaggle/working/runs/detect/train/weights/best.pt"
INPUT_VIDEO = "/kaggle/input/video-coca-cola/Coca-Cola _ For Everyone _30.mp4"
OUTPUT_DIR = Path("/kaggle/working/processed-videos")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO_1 = str(OUTPUT_DIR / "method1_automatic.mp4")
OUTPUT_VIDEO_2 = str(OUTPUT_DIR / "method2_manual.mp4")
OUTPUT_VIDEO_3 = str(OUTPUT_DIR / "method3_statistics.mp4")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
LINE_WIDTH = 2
FONT_SCALE = 0.5
SHOW_LABELS = True
SHOW_CONF = True

def process_video_simple():
    print("=" * 50)
    print("METHOD 1: Automatic Processing")
    print("=" * 50)
    
    model = YOLO(MODEL_PATH)
    
    results = model.predict(
        source=INPUT_VIDEO,
        save=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        show_labels=SHOW_LABELS,
        show_conf=SHOW_CONF,
        line_width=LINE_WIDTH,
        project=str(OUTPUT_DIR),
        name="temp_method1",
        verbose=True
    )
    
    import shutil
    temp_output = OUTPUT_DIR / "temp_method1" / Path(INPUT_VIDEO).name
    if temp_output.exists():
        shutil.move(str(temp_output), OUTPUT_VIDEO_1)
        shutil.rmtree(OUTPUT_DIR / "temp_method1")
    
    print(f"Video saved: {OUTPUT_VIDEO_1}\n")
    return OUTPUT_VIDEO_1

def process_video_manual():
    print("=" * 50)
    print("METHOD 2: Manual Processing")
    print("=" * 50)
    
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {INPUT_VIDEO}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_2, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    
    print(f"Video parameters: {width}x{height}, {fps} fps, total frames: {total_frames}\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        annotated_frame = results[0].plot(
            line_width=LINE_WIDTH,
            font_size=FONT_SCALE,
            labels=SHOW_LABELS,
            conf=SHOW_CONF
        )
        
        if len(results[0].boxes) > 0:
            detection_count += 1
        
        out.write(annotated_frame)
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Processed: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    print("\n" + "=" * 50)
    print("Processing completed!")
    print("=" * 50)
    print(f"Processed frames: {frame_count}")
    print(f"Frames with detections: {detection_count}")
    print(f"Result saved: {OUTPUT_VIDEO_2}\n")
    
    return OUTPUT_VIDEO_2

def process_video_with_stats():
    print("=" * 50)
    print("METHOD 3: With Detailed Statistics")
    print("=" * 50)
    
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {INPUT_VIDEO}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_3, fourcc, fps, (width, height))
    
    class_counts = {}
    frame_count = 0
    
    print(f"Processing video: {width}x{height} @ {fps}fps\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        
        annotated_frame = results[0].plot(line_width=LINE_WIDTH)
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        out.write(annotated_frame)
        
        if frame_count % 50 == 0:
            print(f"  Frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    print("\n" + "=" * 50)
    print("Detection Statistics:")
    print("=" * 50)
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} detections")
    
    print(f"\nVideo saved: {OUTPUT_VIDEO_3}\n")
    
    return OUTPUT_VIDEO_3

if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH}")
        exit(1)
    
    if not Path(INPUT_VIDEO).exists():
        print(f"Video not found: {INPUT_VIDEO}")
        exit(1)
    
    print("Processing all methods...\n")
    
    video1 = process_video_simple()
    video2 = process_video_manual()
    video3 = process_video_with_stats()
    
    print("\n" + "=" * 50)
    print("ALL PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"\nAll videos saved in: {OUTPUT_DIR}")
    print(f"1. {OUTPUT_VIDEO_1}")
    print(f"2. {OUTPUT_VIDEO_2}")
    print(f"3. {OUTPUT_VIDEO_3}")