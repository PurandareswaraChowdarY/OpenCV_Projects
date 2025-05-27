import cv2
import sys
import time
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def select_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        return cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        return cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise ValueError('Unsupported tracker type')

def detect_scene_change(prev_frame, curr_frame, ssim_threshold=0.5):
    grayA = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    grayA = cv2.resize(grayA, (320, 240))
    grayB = cv2.resize(grayB, (320, 240))
    ssim_score = compare_ssim(grayA, grayB)
    return ssim_score < ssim_threshold

def resize_frame(frame, width=640, height=480):
    return cv2.resize(frame, (width, height))

def main():
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
    print("Select tracker type:")
    for i, t_type in enumerate(tracker_types, start=1):
        print(f"{i}. {t_type}")

    tracker_choice = int(input("Enter tracker number: ")) - 1
    tracker_type = tracker_types[tracker_choice]
    tracker = select_tracker(tracker_type)

    video_path = input("Enter video file path (leave blank for webcam): ")
    if not video_path:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print("Error: Could not read video file.")
        sys.exit()

    frame = resize_frame(frame)
    bbox = cv2.selectROI("Select ROI", frame, False)
    ok = tracker.init(frame, bbox)
    if not ok:
        print("Error: Tracker initialization failed.")
        sys.exit()

    prev_frame = frame.copy()

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 640, 480)

    prev_time = time.time()

    while True:
        ok, frame = video.read()
        if not ok:
            break

        frame = resize_frame(frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Scene cut detection
        if detect_scene_change(prev_frame, frame):
            print("Scene cut detected — reinitializing tracker")
            bbox = cv2.selectROI("Select ROI", frame, False)
            tracker = select_tracker(tracker_type)
            ok = tracker.init(frame, bbox)
            if not ok:
                print("Error: Tracker re-initialization failed after scene cut.")
                break

        prev_frame = frame.copy()

        ok, bbox = tracker.update(frame)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame, f"{tracker_type} Tracker", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "ESC to quit", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # FPS counter display
        cv2.putText(frame, f"FPS: {fps:.2f}", (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
