import cv2

cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("❌ Failed to open video!")
else:
    print("✅ Video opened successfully!")

cap.release()