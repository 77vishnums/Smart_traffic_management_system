# Smart_traffic_management_system

Project Description

This project, titled "Smart Traffic System," is an intelligent traffic management solution designed to reduce urban congestion. Instead of using fixed timers, the system dynamically adjusts the duration of green lights at an intersection based on the real-time traffic density.

It uses a popular computer vision model called **YOLO (You Only Look Once)** to analyze images from different directions of a traffic junction (North, East, South, and West). By detecting and counting the number of vehicles (cars, motorcycles, buses, trucks) in each direction, the system can intelligently allocate more green-light time to the lanes with heavier traffic. This adaptive approach aims to optimize traffic flow, reduce waiting times, lower fuel consumption, and improve the overall efficiency of the intersection.

The Python script you've provided is the core implementation of this logic. It processes static images for each direction to simulate this real-time decision-making process. 

### How the Code Works

The Python script uses the OpenCV library for image processing and the pre-trained YOLOv3 model for object detection. Here’s a step-by-step explanation:

1.  Loading the YOLO Model (`load_yolo_model` function)
     `cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")`: This line is the first crucial step. It loads the YOLOv3 model into memory. The model consists of two files:
         `yolov3.weights`: This file contains the pre-trained weights of the neural network. It's the "brain" of the model, containing all the learned patterns for recognizing objects.
         `yolov3.cfg`: This is the configuration file that defines the architecture of the neural network—how many layers it has, what kind of layers they are, etc.
     `net.getUnconnectedOutLayers()`: YOLO's architecture has several layers. This line identifies the final output layers, which are the ones that provide the object detection results.

2.  Counting Vehicles in an Image (`count_vehicles` function)
   `blob = cv2.dnn.blobFromImage(...)`: An image is not fed directly into the neural network. It must be pre-processed into a format the network understands, called a "blob." This function resizes and normalizes the image to match the input requirements of YOLO 
    * `net.setInput(blob)` and `outputs = net.forward(output_layers)`: These lines perform the main task. The processed image (blob) is passed through the YOLO network, which then outputs its predictions about objects found in the image.
    Looping Through Detections: The code then iterates through all the `outputs` from the model.
         `class_id = np.argmax(scores)` and `confidence = scores[class_id]`: For each detected object, the code checks its `confidence` score (how sure the model is) and determines its `class_id` .
         `if confidence > 0.5 and class_id in [2, 3, 5, 7]`: This is the filtering step. It only considers a detection valid if the model is more than 50% confident. Crucially, it checks if the detected object's `class_id` corresponds to a vehicle (`2` for car, `3` for motorcycle, `5` for bus, `7` for truck in the COCO dataset which YOLOv3 was trained on).
       `vehicle_count += 1`: If the detection is a confident vehicle, the counter is incremented.
         `cv2.rectangle(...)` and `cv2.putText(...)`: For visualization, a green bounding box is drawn around each detected vehicle, and its class ID is written above the box.

3.  Main Function (`main`)
     `net, output_layers = load_yolo_model()`: It starts by loading the YOLO model.
     `images = {...}`: It defines a dictionary containing the paths to the four traffic images (`north_traffic.jpeg`, `east_traffic.jpeg`, etc.).
     Processing Each Direction**: The code then enters a loop to process one image at a time.
         `vehicle_count, height, width = count_vehicles(...)`: It calls the function to count vehicles in the current image.
         Dynamic Signal Timing**: This is where the "smart" logic happens.
             `green_light_duration = 10`: A base green light time of 10 seconds is set.
             `if vehicle_count > 40:`: If traffic is very heavy (more than 40 vehicles), it adds 20 seconds.
             `elif vehicle_count > 20:`: If traffic is medium (more than 20 vehicles), it adds 10 seconds.
        Displaying Results**:
             `cv2.putText(...)`: It writes the total vehicle count and the calculated green light duration directly onto the image.
            `cv2.imshow(direction, image)`: It opens a window to display the processed image with all the bounding boxes and text.
            `cv2.waitKey(green_light_duration * 1000)`: This line pauses the program execution for the duration of the calculated green light time (the duration is converted from seconds to milliseconds). This simulates the traffic light being green.
             `cv2.destroyWindow(direction)`: After the pause, the image window is closed, simulating the light turning red.
     The loop continues until all four directions have had their turn, simulating a full traffic light cycle.
