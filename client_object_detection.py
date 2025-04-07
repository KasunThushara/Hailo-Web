#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import queue
import threading
import cv2
from typing import List
from object_detection_utils import ObjectDetectionUtils
import socket
import pickle
import time

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_images_opencv, validate_images, divide_list_to_batches


CAMERA_CAP_WIDTH = 640
CAMERA_CAP_HEIGHT = 480

# Socket setup
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

server_ip = "127.0.0.1"  # Replace with server IP
server_port = 9999
address = (server_ip, server_port)


        
def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example")
    parser.add_argument(
        "-n", "--net", 
        help="Path for the network in HEF format.",
        default="yolov7.hef"
    )
    parser.add_argument(
        "-i", "--input", 
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size", 
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-l", "--labels", 
        default="coco.txt",
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
    )
    parser.add_argument(
        "-s", "--save_stream_output",
        action="store_true",
        help="Save the output of the inference from a stream."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    return args


def preprocess(
    images: List[np.ndarray],
    cap: cv2.VideoCapture,
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
    utils: ObjectDetectionUtils
) -> None:
    """
    Preprocess and enqueue images or camera frames into the input queue as they are ready.

    Args:
        images (List[np.ndarray], optional): List of images as NumPy arrays.
        camera (bool, optional): Boolean indicating whether to use the camera stream.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
    """
    if cap is None:
        preprocess_images(images, batch_size, input_queue, width, height, utils)
    else:
        preprocess_from_cap(cap, batch_size, input_queue, width, height, utils)

    input_queue.put(None)  # Add sentinel value to signal end of input

def preprocess_from_cap(cap: cv2.VideoCapture, batch_size: int, input_queue: queue.Queue, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """
    Process frames from the camera stream and enqueue them.

    Args:
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
    """
    frames = []
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = utils.preprocess(processed_frame, width, height)
        processed_frames.append(processed_frame)

        if len(frames) == batch_size:
            input_queue.put((frames, processed_frames))
            processed_frames, frames = [], []


def preprocess_images(images: List[np.ndarray], batch_size: int, input_queue: queue.Queue, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """
    Process a list of images and enqueue them.

    Args:
        images (List[np.ndarray]): List of images as NumPy arrays.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
    """
    for batch in divide_list_to_batches(images, batch_size):
        input_tuple = ([image for image in batch], [utils.preprocess(image, width, height) for image in batch])
        input_queue.put(input_tuple)


def postprocess(
    output_queue: queue.Queue,
    cap: cv2.VideoCapture,
    utils: ObjectDetectionUtils
) -> None:
    """
    Process and print the output results to console.
    """
    # Initialize FPS variables
    prev_time = 0
    curr_time = 0
    
    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received

        original_frame, infer_results = result

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Deals with the expanded results from hailort versions < 4.19.0
        if len(infer_results) == 1:
            infer_results = infer_results[0]

        detections = utils.extract_detections(infer_results)

        # Draw detections
        frame_with_detections = utils.draw_detections(detections, original_frame)
        
        # Display FPS on frame
        cv2.putText(frame_with_detections, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Send frame over socket
        ret, buffer = cv2.imencode(".jpg", frame_with_detections, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        data = pickle.dumps(buffer)
        client_socket.sendto(data, address)
        
    output_queue.task_done()  # Indicate that processing is complete

def infer(
    input,
    net_path: str,
    labels_path: str,
    batch_size: int,
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        images (List[Image.Image]): List of images to process.
        net_path (str): Path to the HEF model file.
        labels_path (str): Path to a text file containing labels.
        batch_size (int): Number of images per batch.
        output_path (Path): Path to save the output images.
    """
    det_utils = ObjectDetectionUtils(labels_path)
    
    cap = None
    images = []
    if input == "camera":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)
    elif any(input.lower().endswith(suffix) for suffix in ['.mp4', '.avi', '.mov', '.mkv'] ):
        cap = cv2.VideoCapture(input)
    else:
        images = load_images_opencv(input)

        # Validate images
        try:
            validate_images(images, batch_size)
        except ValueError as e:
            logger.error(e)
            return

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size, send_original_frame=True
    )
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(images, cap, batch_size, input_queue, width, height, det_utils)
    )
    postprocess_thread = threading.Thread(
        target=postprocess,
        args=(output_queue, cap, det_utils)
    )

    preprocess_thread.start()
    postprocess_thread.start()

    hailo_inference.run()
    
    preprocess_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()

    logger.info('Inference was successful!')


def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()

    # Start the inference
    infer(args.input, args.net, args.labels, args.batch_size)


if __name__ == "__main__":
    main()
