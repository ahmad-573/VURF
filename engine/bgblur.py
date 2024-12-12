import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get some video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Frame count:", frame_count)
    print("FPS:", fps)
    print("Resolution:", (width, height))

    frames_needed = [20,40,60,80,100]
    # Read and save each frame
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame", i)
            break
        
        # Save frame
        if i in frames_needed:
            frame_filename = os.path.join(output_folder, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_filename, frame)

    # Release the video capture object
    cap.release()
    print("Frames extraction completed.")

if __name__ == "__main__":
    video_path = "g_run.mp4"  # Change this to your video file path
    output_folder = "frames"  # Change this to the desired output folder

    extract_frames(video_path, output_folder)
