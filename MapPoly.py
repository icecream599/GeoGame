import cv2
import numpy as np
import pickle
import cvzone

MAP_IMAGE_PATH = 'darkmap.png'  # Video or static image of map 'darkmap.png'
COUNTRIES_FILE = 'countries.p'
WIDTH, HEIGHT = 1280, 720
DETECTION_MODE = True
POLYGON_MODE = False
current_polygon = []
polygons = []

try:
    with open(COUNTRIES_FILE, 'rb') as f:
        polygons = pickle.load(f)
        print(f"Loaded {len(polygons)} countries.")
except FileNotFoundError:
    print("No saved countries file found. Starting fresh.")
    polygons = []

cap = cv2.VideoCapture(MAP_IMAGE_PATH)
cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Map", 900, 600)


def find_map_corners(img):
    """
    Finds the corners of the map using cvzone.findContours, filtering for quadrilaterals.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 30, 80)


    imgContours, conFound = cvzone.findContours(img, imgCanny,filter=[4])

    if conFound:
        largest_contour = conFound[0]['cnt']
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return np.array([])


def warp_image(img, points, size=(WIDTH, HEIGHT)):
    """
    Warps the image based on the detected corner points.
    """
    points = sorted(points, key=lambda x: x[1])
    if points[0][0] > points[1][0]:
        points[0], points[1] = points[1], points[0]
    if points[2][0] > points[3][0]:
        points[2], points[3] = points[3], points[2]

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarped = cv2.warpPerspective(img, matrix, size)
    return imgWarped,matrix


def mousePoints(event, x, y, flags, params):
    """
    Captures mouse clicks to define the vertices of a polygon.
    """
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN and POLYGON_MODE:
        current_polygon.append((x, y))


cv2.setMouseCallback("Map", mousePoints)
map_corners = None
warped = None

while True:
    if DETECTION_MODE:
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video if it ends
            continue
        display = img.copy()
    else:
        display = warped.copy()

    if DETECTION_MODE:
        corners = find_map_corners(img)
        if corners.size != 0:
            cv2.drawContours(display, [corners], -1, (0, 255, 0), 5)
            cvzone.putTextRect(display, "Press 'c' to confirm map", (50, 50), scale=2, thickness=2, colorR=(0, 255, 0))
        else:
            cvzone.putTextRect(display, "Map not detected", (50, 50), scale=2, thickness=2, colorR=(0, 0, 255))
    else:  # POLYGON_MODE
        # Draw saved country polygons
        overlay = display.copy()
        for poly, name in polygons:
            cv2.polylines(display, [np.array(poly)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(overlay, [np.array(poly)], (0, 255, 0))
            (x, y), _ = cv2.minEnclosingCircle(np.array(poly))
            cv2.putText(display, name, (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)

        # Draw current polygon being created
        if current_polygon:
            cv2.polylines(display, [np.array(current_polygon)], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.imshow("Map", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        with open(COUNTRIES_FILE, 'wb') as f:
            pickle.dump(polygons, f)
        print("Saved polygons.")
        break

    elif key == ord('c') and DETECTION_MODE:
        corners = find_map_corners(img)
        if corners.size != 0:
            map_corners = corners
            warped,_ = warp_image(img, map_corners)
            DETECTION_MODE = False
            POLYGON_MODE = True
            print("Map confirmed! Switched to Polygon Mode.")

    elif key == ord('d'):
        DETECTION_MODE = True
        POLYGON_MODE = False
        current_polygon = []
        print("Switched to detection mode.")

    elif key == ord('s') and POLYGON_MODE and len(current_polygon) >= 3:
        name = input("Enter country name: ")
        if name:
            polygons.append([current_polygon.copy(), name])
            current_polygon = []
            print(f"Saved: {name} ({len(polygons)} total)")
        else:
            print("Country name cannot be empty.")

    elif key == ord('x') and polygons:
        removed = polygons.pop()
        current_polygon = []
        print(f"Removed {removed[1]}")

cap.release()
cv2.destroyAllWindows()