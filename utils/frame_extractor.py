import cv2
import os

class FrameExtractor:
    """
    Utility to extract specific frames from video files for analysis.
    """
    @staticmethod
    def extract_frames(video_path, n_frames=1, every_n_frames=None):
        """
        Extracts frames from a video.
        :param video_path: Path to the video file
        :param n_frames: Total number of frames to extract (distributed evenly)
        :param every_n_frames: Extract every Nth frame (overrides n_frames)
        :return: List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if every_n_frames:
            indices = list(range(0, total_frames, every_n_frames))
        else:
            # Distribute n_frames evenly across the video
            indices = [int(i * total_frames / n_frames) for i in range(n_frames)]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        
        cap.release()
        return frames

if __name__ == "__main__":
    print("FrameExtractor module loaded.")
