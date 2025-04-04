#!/usr/bin/env python3

import os
import sys
import argparse
import multiprocessing as mp
from multiprocessing import Process, Event
from loguru import logger
from PIL import Image
import numpy as np
import cv2
from hailo_platform import HEF
from pose_estimation_utils import (output_data_type2dict,
                                 check_process_errors, PoseEstPostProcessing)
import socket
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference

# Socket setup
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)


server_ip = "127.0.0.1"  # Replace with server IP
server_port = 9999
address = (server_ip, server_port)


def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Running pose estimation with USB camera or video file using Hailo API"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input source - either camera index (e.g., 0) or video file path"
    )
    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        default="yolov8s_pose.hef"
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=1,
        type=int,
        help="Number of frames in one batch. Defaults to 1"
    )
    parser.add_argument(
        "-cn", "--class_num",
        help="The number of classes the model is trained on. Defaults to 1",
        default=1
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show real-time visualization"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS for processing. Defaults to 30"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    return args

def is_camera_input(input_str: str) -> bool:
    """Check if input is a camera index (integer)."""
    return input_str.isdigit()

def capture_and_preprocess(
    input_source,
    input_queue: mp.Queue,
    width: int,
    height: int,
    post_processing: PoseEstPostProcessing,
    stop_event: Event,
    target_fps: int = 30
):
    """Combined capture and preprocessing for both camera and video."""
    try:
        # Determine if input is camera or video file
        if is_camera_input(input_source):
            # Camera input
            cap = cv2.VideoCapture(int(input_source))
            if not cap.isOpened():
                logger.error(f"Cannot open camera {input_source}")
                return
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # Video file input
            if not os.path.exists(input_source):
                logger.error(f"Video file not found: {input_source}")
                return
            
            cap = cv2.VideoCapture(input_source)
            if not cap.isOpened():
                logger.error(f"Cannot open video file {input_source}")
                return
            
            # Get video FPS and use it if higher than target FPS
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0 and video_fps < target_fps:
                target_fps = video_fps

        frame_count = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream" if not is_camera_input(input_source) else "Camera disconnected")
                break
            
            frame_count += 1
            
            # Skip frames to match target FPS for video files
            if not is_camera_input(input_source) and target_fps > 0:
                current_fps = cap.get(cv2.CAP_PROP_FPS)
                if current_fps > target_fps:
                    skip_frames = int(current_fps / target_fps) - 1
                    for _ in range(skip_frames):
                        if stop_event.is_set():
                            break
                        cap.grab()
                        frame_count += 1
            
            # Convert to PIL Image and preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            processed_frame = post_processing.preprocess(pil_image, width, height)
            
            # Put processed frame into input queue
            if not input_queue.full():
                input_queue.put([processed_frame])
            else:
                try:
                    input_queue.get_nowait()  # Remove oldest frame
                    input_queue.put([processed_frame])
                except:
                    pass
    except Exception as e:
        logger.error(f"Capture/preprocess error: {e}")
    finally:
        cap.release()
        input_queue.put(None)  # Sentinel value
        logger.info(f"Processed {frame_count} frames")

def postprocess_output(
    output_queue: mp.Queue,
    width: int,
    height: int,
    class_num: int,
    post_processing: PoseEstPostProcessing,
    show: bool,
    stop_event: Event
):
    """Process and visualize the output results."""
    try:
        while not stop_event.is_set():
            try:
                result = output_queue.get(timeout=0.1)
                if result is None:
                    break

                processed_image, raw_detections = result
                predictions = post_processing.post_process(raw_detections, height, width, class_num)
                
                # Convert PIL Image back to OpenCV format (BGR)
                if isinstance(processed_image, Image.Image):
                    processed_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
                
                # Visualize (now working with BGR)
                output_image = post_processing.visualize_pose_estimation_result(
                    predictions, 
                    processed_image
                )
                
                # Crop the padding from the image
                # Get original dimensions (before padding was added)
                h, w = output_image.shape[:2]
                # Calculate the scaling factor used during preprocessing
                scale = min(width / w, height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                # Calculate padding offsets
                pad_x = (width - new_w) // 2
                pad_y = (height - new_h) // 2
                # Crop the image to remove padding
                if pad_y > 0 or pad_x > 0:
                    output_image = output_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w]
                
                # Encode and send frame via UDP
                ret, buffer = cv2.imencode(".jpg", output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if ret:
                    try:
                        data = pickle.dumps(buffer)
                        client_socket.sendto(data, address)
                    except Exception as e:
                        logger.error(f"Socket error: {e}")
                
                # Optional: Local display if show flag is set
                if show:
                    cv2.imshow('Pose Estimation', output_image)
                    if cv2.waitKey(1) == ord('q'):
                        stop_event.set()
                        break
                        
            except mp.queues.Empty:
                continue
    except Exception as e:
        logger.error(f"Postprocess error: {e}")
    finally:
        if show:
            cv2.destroyAllWindows()
        client_socket.close()
def postprocess_output(
    output_queue: mp.Queue,
    width: int,
    height: int,
    class_num: int,
    post_processing: PoseEstPostProcessing,
    show: bool,
    stop_event: Event
):
    """Process and visualize the output results."""
    try:
        while not stop_event.is_set():
            try:
                result = output_queue.get(timeout=0.1)
                if result is None:
                    break

                processed_image, raw_detections = result
                predictions = post_processing.post_process(raw_detections, height, width, class_num)
                
                # Convert PIL Image back to OpenCV format (BGR)
                if isinstance(processed_image, Image.Image):
                    processed_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
                
                # Visualize (now working with BGR)
                output_image = post_processing.visualize_pose_estimation_result(
                    predictions, 
                    processed_image
                )
                
                # Crop the padding from the image
                # Get original dimensions (before padding was added)
                h, w = output_image.shape[:2]
                # Calculate the scaling factor used during preprocessing
                scale = min(width / w, height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                # Calculate padding offsets
                pad_x = (width - new_w) // 2
                pad_y = (height - new_h) // 2
                # Crop the image to remove padding
                if pad_y > 0 or pad_x > 0:
                    output_image = output_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w]
                
                # Encode and send frame via UDP
                ret, buffer = cv2.imencode(".jpg", output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if ret:
                    try:
                        data = pickle.dumps(buffer)
                        client_socket.sendto(data, address)
                    except Exception as e:
                        logger.error(f"Socket error: {e}")
                
                # Optional: Local display if show flag is set
                if show:
                    cv2.imshow('Pose Estimation', output_image)
                    if cv2.waitKey(1) == ord('q'):
                        stop_event.set()
                        break
                        
            except mp.queues.Empty:
                continue
    except Exception as e:
        logger.error(f"Postprocess error: {e}")
    finally:
        if show:
            cv2.destroyAllWindows()
        client_socket.close()
def infer(
    net_path: str,
    input_source: str,
    batch_size: int,
    class_num: int,
    data_type_dict: dict,
    post_processing: PoseEstPostProcessing,
    show: bool,
    target_fps: int
):
    """Run inference with either camera or video input."""
    stop_event = Event()
    input_queue = mp.Queue(maxsize=2)
    output_queue = mp.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size, output_type=data_type_dict
    )
    height, width, _ = hailo_inference.get_input_shape()

    # Create processes
    capture_process = Process(
        target=capture_and_preprocess,
        name="capture_preprocess",
        args=(input_source, input_queue, width, height, 
             post_processing, stop_event, target_fps)
    )
    
    postprocess_process = Process(
        target=postprocess_output,
        name="postprocessor",
        args=(output_queue, width, height, class_num, 
             post_processing, show, stop_event)
    )

    try:
        capture_process.start()
        postprocess_process.start()
        hailo_inference.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Inference error: {e}")
    finally:
        stop_event.set()
        
        # Clean up queues
        while not input_queue.empty():
            input_queue.get()
        while not output_queue.empty():
            output_queue.get()
        
        # Terminate processes if they're still alive
        if capture_process.is_alive():
            capture_process.terminate()
        if postprocess_process.is_alive():
            postprocess_process.terminate()
        
        capture_process.join()
        postprocess_process.join()
        
        check_process_errors(capture_process, postprocess_process)
        logger.info('Inference completed successfully!')
        
def cleanup():
    """Cleanup resources before exit"""
    if client_socket:
        client_socket.close()
    cv2.destroyAllWindows()
    
def main():
    args = parse_args()
    
    # Register cleanup handler
    import atexit
    atexit.register(cleanup)
    
    output_type_dict = output_data_type2dict(HEF(args.net), 'FLOAT32')
    post_processing = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )

    try:
        infer(
            net_path=args.net,
            input_source=args.input,
            batch_size=args.batch_size,
            class_num=args.class_num,
            data_type_dict=output_type_dict,
            post_processing=post_processing,
            show=args.show,
            target_fps=args.fps
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    finally:
        cleanup()

if __name__ == "__main__":
    main()

