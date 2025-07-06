import pickle
import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time

######################################
cam_id = 1
# width, height = 1920, 1080
# map_file_path = "../Step1_GetCornerPoints/map.p" #no
countries_file_path = "countries.p"
######################################

# file_obj = open(map_file_path, 'rb')
# map_points = pickle.load(file_obj)
# file_obj.close()
print(f"Loaded map coordinates.")
WIDTH, HEIGHT = 1280, 720


# Load previously defined Regions of Interest (ROIs) polygons from a file
if countries_file_path:
    file_obj = open(countries_file_path, 'rb')
    polygons = pickle.load(file_obj)
    file_obj.close()
    print(f"Loaded {len(polygons)} countries.")
    print(f'polygons {polygons}')
else:
    polygons = []

cap = cv2.VideoCapture(cam_id)
# Set the width and height of the webcam frame
# cap.set(3, width)
# cap.set(4, height)

cv2.namedWindow("Stacked Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stacked Image", 900, 600)

counter = 0

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False,
                        maxHands=1,
                        modelComplexity=1,
                        detectionCon=0.5,
                        minTrackCon=0.5)

questions = [["Point to the continent with the Sahara Desert", "Africa"],
             ["Point to the continent with the smallest countries", "Europe"],
             ["which country is a continent?", "Australia"],
             ["Point to the continent with the Grand Canyon", "North-America"],
             ["Where can you find the largest river by volume?","South-America"],
             ["Where can you find Mount Everest?","Asia"]
             ]

selected_country = None
country_entry_times = {}

counter_country = 0
counter_answer = 0
current_question = 0
start_counter = False

answer_color = (0, 0, 255)
total_score = 0

DETECTION_MODE = True
POLYGON_MODE = False



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
        # Don't judge me, i had to repeat this because it wasn't working with the cvzone
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return np.array([])


def warp_image(img, points, size=(WIDTH, HEIGHT)):
    """
    Warps the image based on the detected corner points.
    """
    # Sort by y-coordinate
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


def warp_single_point(point, matrix):
    """
    Warp a single point using the provided perspective transformation matrix.

    Parameters:
    - point: Coordinates of the point to be warped.
    - matrix: Perspective transformation matrix.

    Returns:
    - point_warped: Warped coordinates of the point.
    """
    # convert the point to homogeneous coordinates
    point_homogeneous = np.array([[point[0], point[1], 1]], dtype=np.float32)
    point_homogeneous_transformed = np.dot(matrix, point_homogeneous.T).T
    # convert back to non-homogeneous coordinates
    point_warped = point_homogeneous_transformed[0, :2] / point_homogeneous_transformed[0, 2]

    print(f'point warped {point_warped}')
    return point_warped


def get_finger_location(img, imgWarped):
    """
    Get the location of the index finger tip in the warped image.

    Parameters:
    - img: Original

 image.

    Returns:
    - warped_point: Coordinates of the index finger tip in the warped image.
    """
    hands, img = detector.findHands(img, draw=False, flipType=True)
    # Check if any hands are detected
    if hands:
        hand1 = hands[0]  # Get the first hand detected
        indexFinger = hand1["lmList"][8][0:2]
        # cv2.circle(img,indexFinger,5,(255,0,255),cv2.FILLED)
        warped_point = warp_single_point(indexFinger, matrix)
        warped_point = int(warped_point[0]), int(warped_point[1])
        print(indexFinger, warped_point)
        cv2.circle(imgWarped, warped_point, 5, (255, 0, 0), cv2.FILLED)
    else:
        warped_point = None

    return warped_point


def inverse_warp_image(img, imgOverlay, map_points):
    """
    Inverse warp an overlay image onto the original image using provided map points.

    Parameters:
    - img: Original image.
    - imgOverlay: Overlay image to be warped.
    - map_points: List of four points representing the region on the map.

    Returns:
    - result: Combined image with the overlay applied.
    """

    # Sort by y-coordinate
    points = sorted(map_points, key=lambda x: x[1])
    if points[0][0] > points[1][0]:
        points[0], points[1] = points[1], points[0]
    if points[2][0] > points[3][0]:
        points[2], points[3] = points[3], points[2]

    map_points = np.array(points, dtype=np.float32)

    destination_points = np.array([[0, 0], [imgOverlay.shape[1] - 1, 0], [0, imgOverlay.shape[0] - 1],
                                   [imgOverlay.shape[1] - 1, imgOverlay.shape[0] - 1]], dtype=np.float32)

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(destination_points, map_points)

    warped_overlay = cv2.warpPerspective(imgOverlay, M, (img.shape[1], img.shape[0]))

    # Combine the original image with the warped overlay
    result = cv2.addWeighted(img, 1, warped_overlay, 0.65, 0, warped_overlay)

    return result


def create_overlay_image(polygons, warped_point, imgOverlay):
    """
    Create an overlay image with marked polygons based on the warped finger location.
    Fixed to handle multiple polygons with the same name properly.

    Parameters:
    - polygons: List of polygons representing countries.
    - warped_point: Coordinates of the index finger tip in the warped image.
    - imgOverlay: Overlay image to be marked.

    Returns:
    - imgOverlay: Overlay image with marked polygons.
    """

    country_selected = None
    green_duration_threshold = 2.0
    currently_touching = []

    for i, (polygon, name) in enumerate(polygons):
        # Create unique identifier for each polygon
        polygon_id = f"{name}_{i}"

        polygon_np = np.array(polygon, np.int32).reshape((-1, 1, 2))
        result = cv2.pointPolygonTest(polygon_np, warped_point, False)

        if result >= 0:
            currently_touching.append(polygon_id)

            if polygon_id not in country_entry_times:
                country_entry_times[polygon_id] = time.time()

            time_in_polygon = time.time() - country_entry_times[polygon_id]

            if time_in_polygon >= green_duration_threshold:
                color = (0, 255, 0)
                country_selected = name
            else:
                color = (255, 255, 0)
                # Draw an arc around the finger point based on elapsed time
                angle = int((time_in_polygon / green_duration_threshold) * 360)
                cv2.ellipse(imgOverlay, (warped_point[0], warped_point[1] - 100),
                            (50, 50), 0, 0, angle, (0, 255, 0),
                            thickness=-1)

            cv2.polylines(imgOverlay, [np.array(polygon)], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(imgOverlay, [np.array(polygon)], (169, 169, 169))
            cvzone.putTextRect(imgOverlay, name, polygon[0], scale=3, thickness=3, colorR=(255, 255, 0))
            cvzone.putTextRect(imgOverlay, name, (0, 100), scale=5, thickness=5, colorR=(255, 255, 0))

    # Clean up timing data for polygons that are no longer being touched
    polygons_to_remove = []
    for polygon_id in country_entry_times:
        if polygon_id not in currently_touching:
            polygons_to_remove.append(polygon_id)

    for polygon_id in polygons_to_remove:
        country_entry_times.pop(polygon_id, None)

    return imgOverlay, country_selected


def check_answer(name, current_question, img, total_score):
    global counter_answer, start_counter, answer_color

    # Check if game is completed
    if current_question >= len(questions):
        # Display final score and restart instruction
        cvzone.putTextRect(img, f"GAME COMPLETE!", (100, 150), scale=3, thickness=5, colorR=(0, 255, 0))
        cvzone.putTextRect(img, f"Your score: {total_score}/{len(questions)}", (100, 200), scale=2, thickness=3,
                           colorR=(255, 255, 0))
        cvzone.putTextRect(img, f"Press 'r' to restart", (100, 250), scale=2, thickness=3, colorR=(255, 255, 0))
        return current_question, total_score

    # Display current question and score
    cvzone.putTextRect(img, f"Question {current_question + 1}/{len(questions)}", (10, 30), scale=1.5, thickness=2,
                       colorR=(255, 255, 0))
    cvzone.putTextRect(img, f"Score: {total_score}/{current_question}", (10, 70), scale=1.5, thickness=2,
                       colorR=(255, 255, 0))

    if name != None:
        if name == questions[current_question][1]:
            start_counter = 'CORRECT'
            answer_color = (0, 255, 0)
        else:
            start_counter = 'WRONG'
            answer_color = (0, 0, 255)

    if start_counter:
        counter_answer += 1
        if counter_answer != 0:
            cvzone.putTextRect(img, start_counter, (300, 300), colorR=answer_color)
        if counter_answer == 70:
            counter_answer = 0
            current_question += 1
            if start_counter == "CORRECT":
                total_score += 1
            start_counter = False

    return current_question, total_score


# reset game
def reset_game():
    """
    Reset all game variables to start a new game
    """
    global counter_country, counter_answer, current_question, start_counter
    global answer_color, total_score, country_entry_times, selected_country

    counter_country = 0
    counter_answer = 0
    current_question = 0
    start_counter = False
    answer_color = (255, 255, 0)
    total_score = 0
    country_entry_times = {}
    selected_country = None

    print("Game restarted!")


def show_welcome_message(img):
    """
    Display welcome message and instructions at the start
    """
    if current_question == 0 and total_score == 0 and not start_counter:
        cvzone.putTextRect(img, "Welcome to Geography Quiz!", (100, 100), scale=2, thickness=3, colorR=answer_color)
        cvzone.putTextRect(img, "Point to countries with your finger", (100, 150), scale=1.5, thickness=2, colorR=answer_color)
        cvzone.putTextRect(img, "Hold for 2 seconds to select", (100, 180), scale=1.5, thickness=2, colorR=answer_color)
        cvzone.putTextRect(img, "Press 'r' anytime to restart", (100, 210), scale=1.5, thickness=2, colorR=answer_color)



warped = None
map_corners = None
# imgOverlay = None
# imgOutput = None

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    imgOverlay = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video if it ends
        continue

    # if DETECTION_MODE:
    #
    #     display = img.copy()
    # else:
    #     # In polygon mode, we work with the static warped image
    #     # and a fresh copy of it for displaying polygons without overlap issues
    #     warped,_ = warp_image(img, map_corners)
    #     disp = warped.copy()
    #     display = warped.copy()

    if DETECTION_MODE:
        display = img.copy()
        corners = find_map_corners(img)
        if corners.size != 0:
            map_corners = corners
            cv2.drawContours(display, [corners], -1, (0, 255, 0), 5)
            cvzone.putTextRect(display, "Press 'c' to confirm map", (50, 50), scale=2, thickness=2, colorR=(0, 255, 0))
        else:
            cvzone.putTextRect(display, "Map not detected", (50, 50), scale=2, thickness=2, colorR=(0, 0, 255))
    else:
        warped, matrix = warp_image(img, map_corners)
        disp = warped.copy()
        display = warped.copy()
        # normal working code

        # Read a frame from the webcam
        # success, img = cap.read()
        # imgWarped, matrix = warp_image(img, map_corners)
        # imgWarped = warpImg.copy()
        imgOutput = img.copy()

        # Find the hand and its landmarks
        warped_point = get_finger_location(img, warped)

        # h, w, _ = warped.shape
        # imgOverlay = np.zeros((h, w, 3), dtype=np.uint8)

        selected_country = None
        if warped_point:
            imgOverlay, selected_country = create_overlay_image(polygons, warped_point, imgOverlay)
            imgOutput = inverse_warp_image(img, imgOverlay, map_corners)
            print(f'selected country {selected_country}')

        # Display the current question
        if current_question != len(questions):
            cvzone.putTextRect(imgOutput, questions[current_question][0], (0, 100),1.5,colorR=(255,255,0))
        current_question, total_score = check_answer(selected_country, current_question, imgOutput, total_score)

        show_welcome_message(img)

    imgStacked = cvzone.stackImages([img,display,imgOutput,imgOverlay], 2, 1)
    cv2.imshow("Stacked Image", imgStacked)
    # cv2.imshow("Stacked Image", imgOutput)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        with open(countries_file_path, 'wb') as f:
            pickle.dump(polygons, f)
        print("Saved polygons.")
        break

    elif key == ord('c') and DETECTION_MODE:
        corners = find_map_corners(img)
        if corners.size != 0:
            map_corners = corners
            warped,matrix = warp_image(img, map_corners)  # Create the warped image once
            DETECTION_MODE = False
            POLYGON_MODE = True
            print("Map confirmed! Switched to Polygon Mode.")

    elif key == ord('d'):
        DETECTION_MODE = True
        POLYGON_MODE = False
        current_polygon = []
        print("Switched to detection mode.")

    elif key == ord('r'):  # Restart
        reset_game()
        print("Game restarted! Starting from question 1.")

    elif key == ord('s'):  # Skip current question (optional)
        if not DETECTION_MODE and current_question < len(questions):
            current_question += 1
            start_counter = False
            counter_answer = 0
            print(f"Skipped to question {current_question + 1}")

    # cv2.imshow("Original Image", img)
    # cv2.imshow("Warped Image", imgWarped)

    # cv2.imshow("Output Image", imgOutput)
    # key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()