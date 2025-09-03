import cv2 # type: ignore
import numpy as np # type: ignore
import os
import time

# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]
    return net, output_layers

# Count vehicles in an image and draw bounding boxes
def count_vehicles(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    vehicle_count = 0
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter for vehicle classes (car, motorcycle, bus, truck)
            if confidence > 0.5 and class_id in [2, 3, 5, 7]:  # 2: car, 3: motorcycle, 5: bus, 7: truck
                vehicle_count += 1
                # Draw bounding box
                x, y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
                cv2.putText(image, f"{class_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return vehicle_count, height, width  # Return height and width along with vehicle count

# Main function to run the vehicle counting on images
def main():
    net, output_layers = load_yolo_model()
    
    # Define image paths (assuming images are in the project directory)
    project_directory = os.path.dirname(os.path.abspath(__file__))
    images = {
       'North': os.path.join(project_directory, 'north_traffic.jpeg'),
        'East': os.path.join(project_directory, 'east_traffic.jpeg'),
        'South': os.path.join(project_directory, 'south_traffic.jpeg'),
        'West': os.path.join(project_directory, 'west_traffic.jpeg')
    }

    # Process each direction
    for direction, img_path in images.items():
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}. Check if the file exists and is accessible.")
            continue

        vehicle_count, height, width = count_vehicles(image, net, output_layers)  # Get count and dimensions

        # Display vehicle count on the image
        cv2.putText(image, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Adjust signal timing based on vehicle count
        green_light_duration = 10  # Base duration
        if vehicle_count > 40:
            green_light_duration += 20  # Add 20 seconds for heavy traffic
        elif vehicle_count > 20:
            green_light_duration += 10  # Add 10 seconds for medium traffic

        # Draw signal status on the image
        signal_status = f"Green light for {direction}: {green_light_duration} seconds"
        cv2.putText(image, signal_status, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"{direction} light is GREEN.", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the image with bounding boxes and vehicle count
        cv2.imshow(direction, image)
        cv2.waitKey(green_light_duration * 1000)  # Wait for the green light duration
        cv2.destroyWindow(direction)

        print(f"{direction} light is RED.")
        time.sleep(2)  # Wait for 2 seconds before next cycle

if __name__ == "__main__":
    main()
