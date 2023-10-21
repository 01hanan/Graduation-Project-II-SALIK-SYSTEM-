import random
import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
import math
import requests
from supervision.draw.color import Color
from ultralytics import YOLO
import supervision as sv
import argparse
import base64
from tracker import Tracker
import geocoder


# Step 1: Define a function to send SMS with pothole information
def send_sms(cam, image_file, className, conf):

    # Our API key from imgBB
    api_key = '023a2a1fd5cadc3a4dc7f73b891268d6'

    # Read the image file in binary mode, encode it in base64, and decode it as UTF-8
    with open(image_file, 'rb') as file:
        image_data = base64.b64encode(file.read()).decode('utf-8')

     # Define the API endpoint URL
    url = "https://api.imgbb.com/1/upload"

    # Prepare payload for the POST request containing API key and encoded image data
    payload = {
        "key": api_key,
        "image": image_data
    }

    # Send the POST request to upload the image
    response = requests.post(url, data=payload)

    # Parse the JSON response
    data = response.json()

    # Check if the upload was successful
    if response.status_code == 200 and data['success']:

        # Extract the URL of the uploaded image from the response data
        image_url = data['data']['url']
        print(f"Image uploaded successfully. URL: {image_url}")

    else:
        # Handle the case where the image upload failed
        print("Failed to upload the image.")

    # Define the SMS Gateway URL for sending SMS
    sms_gateway_url = "http://REST.GATEWAY.SA/api/SendSMS"

    # Prepare the parameters required for sending the SMS
    params = {
        "api_id": "API71789973116",  # Your API ID
        "api_password": "salik2023CS",  # Your API password
        "sms_type": "T",  # define SMS type ('T' for text)
        "encoding": "T",  # define encoding type ('T' for text)
        "sender_id": "Gateway.sa",  # Sender ID
        "phonenumber": "966507567242",  # Recipient's phone number
        # Specifies the content of the SMS message
        "textmessage": f"A {className} has been detected at {cam}\nWith accuracy {conf}.\nImage File URL {image_url}"
    }

    # Send the SMS using a POST request to the SMS Gateway
    response = requests.post(sms_gateway_url, params=params)


# Step 2: Main function to run the pothole detection, tracking and SMS alert system
def main():

    # Load the YOLO model
    model = YOLO("best-Yolo8l.pt")
    model.fuse()

    # Initialize the object tracker
    tracker = Tracker()
    colors = [(0, 0, 255) for _ in range(10)]  # Red bounding boxes

    # Get the current location based on the IP address
    location = geocoder.ip('me')

    # Access the latitude and longitude
    latitude = location.latlng[0]
    longitude = location.latlng[1]

    # Create a Google Maps link using the extracted latitude and longitude
    google_maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"

    # Class names used by the YOLO model
    classNames = ["Pothole"]

    # Open a video capture stream
    cap = cv2.VideoCapture(2)

    # Check if a GPU is available and set the device accordingly, else use the CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prints a message to indicate whether the GPU or CPU will be used for processing.
    print("Using Device:", device)

    # Define the detection threshold (potholes with detection confidence scores higher
    # than this threshold are considered valid potholes)
    detection_threshold = 0.4

    # Create an empty dictionary to keep track of the last time an SMS was sent for each object
    last_sms_time = {}

    COOLDOWN_PERIOD = 30  # Time period to prevent repeated SMS alerts

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        results = model.predict(frame)

        for result in results:
            # Initialize an empty list to store detected objects in this result
            detections = []

            for r in result.boxes.data.tolist():

                # Extract bounding box coordinates, confidence score, and class ID
                x1, y1, x2, y2, score, class_id = r

                # Convert coordinates to integers
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                # Ensure class_id is an integer
                class_id = int(class_id)

                # Round the score UP to the nearest integer for display
                score = math.ceil(score * 100) / 100

                # Generate a label for the detected pothole
                labels = [
                    f"{classNames[class_id]} {score}"
                ]
                # Check if the detection score exceeds the specified threshold
                if score > detection_threshold:

                    # Append the details of the detected pothole to the detections list
                    detections.append([x1, y1, x2, y2, score])

             # update the object tracker with the detected pothole.
            tracker.update(frame, detections)

            for track in tracker.tracks:

                # Retrieve the bounding box coordinates and track ID for a tracked pothole
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                # Draw a bounding box around the detected object using its coordinates
                cv2.rectangle(frame, (int(x1), int(y1)), (int(
                    x2), int(y2)), colors[track_id % len(colors)], 2)

                # Extract the class name and confidence score of the detected object
                class_name = classNames[class_id]
                confidence = score

                # Create text to display the class name and confidence score within the bounding box
                text = f"{class_name}: {confidence:.2f}"
                text_x = int(x1)
                text_y = int(y1)

                # Define the background and text colors
                # BGR color code (red in this example)
                background_color = (0, 0, 255)
                # BGR color code (white in this example)
                text_color = (255, 255, 255)

                # Calculate the size of the text and its position within the bounding box
                text_size, _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_w, text_h = text_size

                # Create a filled rectangle as a background for the text
                cv2.rectangle(

                    frame,  # to draw on

                    # the starting and ending coordinates of the rectangle
                    (text_x, text_y - text_h),
                    (text_x + text_w, text_y),
                    background_color,
                    thickness=cv2.FILLED,
                )

                # Display the desired text within the bounding box
                cv2.putText(
                    frame,
                    text,
                    # the position, where the text should be placed
                    org=(text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5,  # font size
                    color=text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,  # line type (AA for anti-aliased )
                    bottomLeftOrigin=False  # to indicate that the origin of the text is at the top left
                )

                # checking if the confidence score of the detected object is greater than or equal to 0.70
                # and if the SMS for this track has not been sent previously.
                if score >= 0.40 and not track.sms_sent:

                    # Get the current time to track the timing of SMS alerts.
                    current_time = time.time()

                    # Retrieve the last time an SMS was sent (default to 0 if not found)
                    last_sms_send_time = last_sms_time.get(track_id, 0)

                    # Check if the time elapsed since the last SMS is greater than the specified cooldown period
                    if current_time - last_sms_send_time >= COOLDOWN_PERIOD:

                        # Mark that an SMS has been sent for this track to avoid duplicate alerts
                        track.sms_sent = True

                        # Update the last SMS send time
                        last_sms_time[track_id] = current_time

                        # Save the current frame as an image file
                        image_filename = "pothole_image.jpg"
                        cv2.imwrite(image_filename, frame)

                        # Send an SMS alert with pothole information to the desired party
                        send_sms(google_maps_link, image_filename,
                                 class_name, confidence)

            # Display the frame with bounding boxes and text
            cv2.imshow("YOLOv8 Detection", frame)

        # Check for the 'Esc' key press to exit the loop
        if (cv2.waitKey(30) == 27):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
