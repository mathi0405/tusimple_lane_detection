import cv2

cap = cv2.VideoCapture("new_test_video.mp4")

if not cap.isOpened():
    print("❌ Could not open video.")
else:
    print("✅ Video opened successfully.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ End of video or read error.")
        break

    cv2.imshow("Video Playback Test", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
