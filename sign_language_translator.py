from jetson_inference import poseNet
import sys
import argparse
import cv2
from jetson_utils import videoSource, videoOutput, cudaToNumpy

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="URI of the input stream")
parser.add_argument("--network", type=str, default="resnet18-hands", help="pre-trained model to load (see below for options)")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use")
args = parser.parse_known_args()[0]

# Load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# Create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

def get_letter_from_pose(pose):
    """
    Determine the letter from the pose keypoints.
    This is a simplified example for demonstration purposes.
    """
    keypoints = pose.Keypoints

    # Simple example for detecting "A" and "B" (you need to expand this)
    if len(keypoints) >= 21:
        thumb_tip = keypoints[4]
        index_tip = keypoints[8]
        middle_tip = keypoints[12]
        ring_tip = keypoints[16]
        pinky_tip = keypoints[20]
        palm_center = keypoints[0]

        if (thumb_tip[1] < palm_center[1] and
            index_tip[1] < palm_center[1] and
            middle_tip[1] < palm_center[1] and
            ring_tip[1] < palm_center[1] and
            pinky_tip[1] < palm_center[1]):
            return "A"

        if (index_tip[1] > palm_center[1] and
            middle_tip[1] > palm_center[1] and
            ring_tip[1] > palm_center[1] and
            pinky_tip[1] > palm_center[1]):
            return "B"

    return "?"

# Capture frames until end-of-stream (or the user exits)
while True:
    image = input.Capture(format='rgb8', timeout=1000)  
    
    if image is None:
        continue

    poses = net.Process(image)
    
    detected_letter = "?"
    for pose in poses:
        detected_letter = get_letter_from_pose(pose)

    # Convert image to numpy array for OpenCV
    img = cudaToNumpy(image)

    # Display detected letter on the image
    cv2.putText(img, detected_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    # Convert back to CUDA image for rendering
    output.Render(image)

    if not input.IsStreaming() or not output.IsStreaming():
        break
