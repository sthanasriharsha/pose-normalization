import os, json, cv2, numpy as np, glob

INPUT_DIR = r"C:\Users\harsh\DWPose\output_normalized"
OUTPUT_DIR = r"C:\Users\harsh\DWPose\pose_frames"

W, H = 512, 512
CONF = 0.3

# body connections
EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13)
]

def draw_pose(coords, scores):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for i, (x,y) in enumerate(coords):
        if scores[i] > CONF:
            cv2.circle(canvas, (int(x), int(y)), 4, (0,255,0), -1)

    for i,j in EDGES:
        if scores[i] > CONF and scores[j] > CONF:
            cv2.line(canvas,
                     (int(coords[i][0]), int(coords[i][1])),
                     (int(coords[j][0]), int(coords[j][1])),
                     (255,255,255), 2)
    return canvas


def process(json_file):
    name = os.path.basename(json_file).replace("_norm_kps.json", "")
    out_folder = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_folder, exist_ok=True)

    with open(json_file) as f:
        data = json.load(f)

    for i, frame in enumerate(data["frames"]):
        coords = np.array(frame["body"]["coords"])
        scores = np.array(frame["body"]["scores"])

        img = draw_pose(coords, scores)

        cv2.imwrite(os.path.join(out_folder, f"{i:05d}.png"), img)

    print("Done:", name)


def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*_norm_kps.json"))
    for f in files:
        process(f)

if __name__ == "__main__":
    main()