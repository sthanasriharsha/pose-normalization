import os
import json
import cv2
import numpy as np
import glob

INPUT_DIR = r"C:\Users\harsh\DWPose\output_normalized"
OUTPUT_DIR = r"C:\Users\harsh\DWPose\avatar"

W, H = 512, 512

# COCO skeleton connections
EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13)
]

CONF = 0.3


def draw_skeleton(frame, coords, scores):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # draw joints
    for i, (x, y) in enumerate(coords):
        if scores[i] > CONF:
            cv2.circle(canvas, (int(x), int(y)), 4, (0,255,0), -1)

    # draw bones
    for (i, j) in EDGES:
        if scores[i] > CONF and scores[j] > CONF:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)

    return canvas


def process(json_path):
    name = os.path.basename(json_path).replace("_norm_kps.json", "")
    print("\nProcessing:", name)

    with open(json_path) as f:
        data = json.load(f)

    fps = int(data.get("fps", 25))
    frames = data["frames"]

    out_path = os.path.join(OUTPUT_DIR, name + "_avatar.mp4")

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )

    for fdata in frames:
        coords = np.array(fdata["body"]["coords"])
        scores = np.array(fdata["body"]["scores"])

        frame = draw_skeleton(None, coords, scores)
        writer.write(frame)

    writer.release()

    print("Saved:", out_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_DIR, "*_norm_kps.json"))

    if not files:
        print("❌ No JSON files found!")
        return

    print(f"Found {len(files)} files")

    for f in files:
        process(f)

    print("\n✅ ALL AVATAR VIDEOS GENERATED")


if __name__ == "__main__":
    main()