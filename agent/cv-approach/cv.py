from PIL import Image
import numpy as np

# === Load grayscale image ===
img = Image.open("VAE-Approach/vae_dataset/frame_00015.png").convert("L")
arr = np.array(img)

# === Define ROI coordinates ===
x_start, y_start = 0, 14
x_end, y_end = 83, 76

#

ai_col = 9
oppo_col = 74


ai_col -= x_start
oppo_col -= x_start

th_up = 150
th_down = 130

# Extract ROI from original image
roi = arr[y_start:y_end + 1, x_start:x_end + 1]  # shape: (63, 84)

max_val = np.max(roi)
coords = np.argwhere(roi == max_val)
ball_pos_roi = coords[0]  # Take the first match
ball_pos_global = (ball_pos_roi[1] + x_start, ball_pos_roi[0] + y_start)

print(f"Ball found at: { tuple(int(v) for v in ball_pos_global)} with intensity {max_val}")


indecies_ai = np.where((roi[:, ai_col] >= th_down) & (roi[:, ai_col] <= th_up))[0]
indecies_oppo = np.where((roi[:, oppo_col] >= th_down) & (roi[:, oppo_col] <= th_up))[0]

ai_loc = int(np.median(indecies_ai) + y_start)
oppo_loc = int(np.median(indecies_oppo) + y_start)

print(f"Agent paddle : {ai_loc} , opponent paddle: {oppo_loc}")