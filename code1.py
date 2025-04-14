import cv2
import numpy as np

weights_path = "yolov4-p6.weights"
config_path = "yolov4-p6.cfg"
labels_path = "coco.names"

with open(labels_path, "r") as f:
    labels = f.read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()

unconnected_out_layers = net.getUnconnectedOutLayers()

if isinstance(unconnected_out_layers, np.ndarray):
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
elif isinstance(unconnected_out_layers, int):
    output_layers = [layer_names[unconnected_out_layers - 1]]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

video_path = "videoplayback.webm"

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(labels[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0) if label == "person" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Video Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()