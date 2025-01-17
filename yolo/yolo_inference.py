from ultralytics import YOLO
from tqdm import tqdm
import cv2

model = YOLO('yolov8x')

# Open the video to get total frame count
video = cv2.VideoCapture('yolo/input_videos/match_short.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video.release()

# Create a tqdm progress bar
progress_bar = tqdm(total=total_frames, desc="Processing frames")

results = model.track('yolo/input_videos/match_short.mp4', conf=0.2, stream=True, save=True)

for result in results:
    print("Boxes:")
    for box in result.boxes:
        print(box)
    
    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

