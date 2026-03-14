import cv2
import glob
import torch
import numpy as np
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# -------------------------
# Load Detection Model
# -------------------------
det_model = YOLO("yolov8n.pt")
det_model.to("cpu")

# -------------------------
# Load Segmentation Model
# -------------------------
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
)

seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
)

seg_model.eval()

# -------------------------
# Cityscapes colors
# -------------------------
colors = np.array([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],
    [190,153,153],[153,153,153],[250,170,30],[220,220,0],
    [107,142,35],[152,251,152],[70,130,180],[220,20,60],
    [255,0,0],[0,0,142],[0,0,70],[0,60,100],
    [0,80,100],[0,0,230],[119,11,32]
])

# -------------------------
# Lane Detection Function
# -------------------------
def detect_lanes(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        50,
        minLineLength=100,
        maxLineGap=50
    )

    lane_img = frame.copy()

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(lane_img,(x1,y1),(x2,y2),(0,255,255),3)

    return lane_img

# -------------------------
# Distance estimation
# -------------------------
def estimate_distance(box_height):

    if box_height == 0:
        return 100

    return round(600 / box_height,2)

# -------------------------
# Load Images
# -------------------------
image_paths = glob.glob(
    r"D:\Chore\codes\leftImg8bit_trainvaltest\leftImg8bit\val\*\*.png"
)

print("Images found:",len(image_paths))

# -------------------------
# Processing Loop
# -------------------------
for img_path in image_paths[:100]:

    frame = cv2.imread(img_path)

    if frame is None:
        continue

    # -------------------------
    # Object Detection
    # -------------------------
    results = det_model(frame)[0]

    det_frame = frame.copy()

    for box in results.boxes:

        x1,y1,x2,y2 = map(int,box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = det_model.names[cls]

        height = y2-y1
        distance = estimate_distance(height)

        # collision warning
        color = (0,255,0)
        warning = ""

        if distance < 6:
            color = (0,0,255)
            warning = "COLLISION RISK"
        elif distance < 10:
            color = (0,165,255)
            warning = "WARNING"

        cv2.rectangle(det_frame,(x1,y1),(x2,y2),color,2)

        text = f"{label} {conf:.2f} {distance}m"

        cv2.putText(
            det_frame,
            text,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        if warning:
            cv2.putText(
                det_frame,
                warning,
                (x1,y2+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

    # -------------------------
    # Semantic Segmentation
    # -------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = seg_model(**inputs)

    logits = outputs.logits

    seg = torch.nn.functional.interpolate(
        logits,
        size=rgb.shape[:2],
        mode="bilinear",
        align_corners=False
    ).argmax(1)[0].cpu().numpy()

    seg_color = np.zeros((seg.shape[0],seg.shape[1],3),dtype=np.uint8)

    for label in range(len(colors)):
        seg_color[seg==label] = colors[label]

    overlay = cv2.addWeighted(frame,0.5,seg_color,0.5,0)

    # -------------------------
    # Lane Detection
    # -------------------------
    lane_frame = detect_lanes(overlay)

    # -------------------------
    # Combine Everything
    # -------------------------
    combined = cv2.addWeighted(det_frame,0.7,lane_frame,0.3,0)

    cv2.imshow("Autonomous Vehicle Perception Stack",combined)

    if cv2.waitKey(500) & 0xFF == 27:
        break

cv2.destroyAllWindows()