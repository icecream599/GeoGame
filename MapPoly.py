import cv2
import numpy as np
import pickle
import cvzone
import os
import atexit

# =====================================================================
# Configuration
DRAWING_ENTITY = "State"
MAP_IMAGE_PATH = 'usa_with_border.jpg'
DATA_FILE = 'usa_states.p'
WIDTH, HEIGHT = 1280, 720
DILATION_ITERATIONS = 1
FLOOD_FILL_TOLERANCE = 10
# =====================================================================

# --- Modes & State Variables ---
DETECTION_MODE = True
MANUAL_MODE, AUTO_MODE = False, False
current_polygon, polygons = [], []
auto_click_point = None

# --- UI Instructions ---
instructions = {
    "detection": "Detection Mode: Press 'c' to confirm map.",
    "auto": "Auto-Find Mode: Click a state. ('m' manual, 'd' reset, 'x' remove, 'q' quit)",
    "manual": "Manual Mode: Click to draw. ('s' save, 'a' auto, 'd' reset, 'x' remove, 'q' quit)"
}


def save_progress():
    if polygons:
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(polygons, f)
        print(f"\nAUTO-SAVED: {len(polygons)} polygons to '{DATA_FILE}'.")
    elif os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        print("\nNo polygons to save. Progress file cleaned up.")


atexit.register(save_progress)

# --- Initial Setup ---
if not os.path.exists(MAP_IMAGE_PATH):
    print(f"ERROR: Map image not found at '{MAP_IMAGE_PATH}'");
    exit()

img_display = cv2.imread(MAP_IMAGE_PATH)
img_processed = img_display.copy()

if img_display is None:
    print(f"ERROR: Could not load map image from '{MAP_IMAGE_PATH}'.");
    exit()

if DILATION_ITERATIONS > 0:
    kernel = np.ones((3, 3), np.uint8)
    img_processed = cv2.dilate(img_processed, kernel, iterations=DILATION_ITERATIONS)

try:
    with open(DATA_FILE, 'rb') as f:
        polygons = pickle.load(f)
        print(f"SUCCESS: Loaded {len(polygons)} {DRAWING_ENTITY}(s) from '{DATA_FILE}'.")
except FileNotFoundError:
    print(f"No data file found at '{DATA_FILE}'. Starting fresh.")

cv2.namedWindow("Map", cv2.WINDOW_NORMAL);
cv2.resizeWindow("Map", 900, 600)


# (Helper functions are unchanged)
def find_state_by_floodfill(img, click_point):
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    lo_diff, up_diff = (FLOOD_FILL_TOLERANCE,) * 3, (FLOOD_FILL_TOLERANCE,) * 3
    cv2.floodFill(img.copy(), mask, click_point, (255, 0, 0), lo_diff, up_diff, cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    contours, _ = cv2.findContours(mask[1:-1, 1:-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return cv2.approxPolyDP(max(contours, key=cv2.contourArea),
                                0.002 * cv2.arcLength(max(contours, key=cv2.contourArea), True), True)
    return None


def find_map_corners(img):
    imgGray, imgBlur, imgCanny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.GaussianBlur(img, (5, 5), 1), cv2.Canny(img,
                                                                                                                    50,
                                                                                                                    150)
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        approx = cv2.approxPolyDP(max(contours, key=cv2.contourArea),
                                  0.02 * cv2.arcLength(max(contours, key=cv2.contourArea), True), True)
        if len(approx) == 4: return approx
    return None


def warp_image(img, points, size=(WIDTH, HEIGHT)):
    pts, rect = points.reshape((4, 2)), np.zeros((4, 2), dtype="float32")
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0], rect[3] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[2] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    matrix = cv2.getPerspectiveTransform(rect, np.array(
        [[0, 0], [size[0] - 1, 0], [0, size[1] - 1], [size[0] - 1, size[1] - 1]], dtype="float32"))
    return cv2.warpPerspective(img, matrix, size)


def mousePoints(event, x, y, flags, params):
    global current_polygon, auto_click_point
    if event == cv2.EVENT_LBUTTONDOWN and AUTO_MODE: auto_click_point = (x, y)


cv2.setMouseCallback("Map", mousePoints)

warped_display, warped_processed = None, None

while True:
    if DETECTION_MODE:
        display = img_display.copy()
        corners = find_map_corners(img_processed)
        if corners is not None: cv2.drawContours(display, [corners], -1, (0, 255, 0), 5)
        cvzone.putTextRect(display, instructions["detection"], (50, 50), scale=1.5)
    else:
        if warped_display is None: break
        display = warped_display.copy()
        mode_text = instructions["auto"]
        cvzone.putTextRect(display, mode_text, (50, 50), scale=1.2, thickness=2, offset=10)
        overlay = display.copy()
        for poly, name in polygons:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            (x, y), _ = cv2.minEnclosingCircle(pts)
            cv2.putText(display, name, (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
        # We don't need manual mode anymore, so this part can be removed for simplicity if desired
        # if MANUAL_MODE and current_polygon: ...

    cv2.imshow("Map", display)

    # --- UPDATED LOGIC FOR IMMEDIATE HIGHLIGHT ---
    if AUTO_MODE and auto_click_point:
        contour = find_state_by_floodfill(warped_processed, auto_click_point)

        if contour is not None:
            # 1. Draw a temporary highlight on the current display image (e.g., in blue)
            cv2.drawContours(display, [contour], -1, (255, 0, 0), 3)  # Blue color for pending

            # 2. Force the window to update NOW to show the highlight
            cv2.imshow("Map", display)
            cv2.waitKey(1)  # This is essential to push the update to the screen

            # 3. Now that the highlight is visible, print to console and wait for input
            print("Contour found!")
            name = input(f"Enter {DRAWING_ENTITY} name: ").strip()

            # 4. Process the name and add the polygon to the main list
            if name:
                polygons.append([[tuple(p[0]) for p in contour], name])
                print(f"Added: {name}")
            else:
                print("Name cannot be empty. Discarding.")

        auto_click_point = None
        # Restart the loop to do a full, clean redraw (shows the new state in permanent green)
        continue

    # Keyboard Input Handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...");
        break
    elif key == ord('c') and DETECTION_MODE:
        corners = find_map_corners(img_processed)
        if corners is not None:
            warped_display = warp_image(img_display, corners)
            warped_processed = warp_image(img_processed, corners)
            DETECTION_MODE, AUTO_MODE = False, True
    # Simplified controls since auto-mode is superior
    elif key == ord('x') and not DETECTION_MODE and polygons:
        removed = polygons.pop()
        print(f"Removed: {removed[1]}")
    elif key == ord('d'):
        DETECTION_MODE, AUTO_MODE = True, False

cv2.destroyAllWindows()