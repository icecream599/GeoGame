import pickle
import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
from enum import Enum
import pygame
import random
import json
import os
import math

# --- 1. Centralized Configuration ---
CAM_ID = 0
WIDTH, HEIGHT = 1280, 720
PAD_RATIO = 0.10

# File Paths
PLAYER_DATA_PATH = "player_data.json"
BACKGROUNDS_DIR = "backgrounds"
CONTINENT_POLYGONS_PATH = "countries.p"
COUNTRY_POLYGONS_PATH = "countries_detailed.p"
CORRECT_SOUND_PATH = "correct.wav"
WRONG_SOUND_PATH = "wrong.wav"

# Post-Processing
ENABLE_POST_PROCESSING = True
CONTRAST_CLIP_LIMIT = 2.0
SHARPEN_INTENSITY = 0.8

# Font and Text Styling
FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_HUD_QUESTION = 1.0
FONT_SCALE_HUD_SCORE = 2.0
FONT_SCALE_FEEDBACK_MAIN = 3.0
FONT_SCALE_FEEDBACK_SUB = 2.0
FONT_SCALE_GAMEOVER_MAIN = 4.0
FONT_SCALE_GAMEOVER_SCORE = 2.5
FONT_SCALE_LEGEND = 1.5
FONT_SCALE_COUNTRY_NAME = 1.5
FONT_SCALE_MENU = 2.5
FONT_SCALE_SHOP = 1.8

# Timing and Game Logic
SELECTION_DELAY_SECONDS = 2.0
FEEDBACK_DISPLAY_FRAMES = 120

# Colors (Defaults)
COLOR_BACKGROUND = (0, 0, 0)
COLOR_CORRECT = (0, 255, 0)  # Default correct color; overridden by equipped highlight.
COLOR_WRONG = (0, 0, 255)
COLOR_NEUTRAL = (255, 255, 0)
COLOR_HOVER = (255, 255, 0)
COLOR_MAP_BORDER = (0, 255, 0)
COLOR_FINGER_TIP = (255, 0, 0)
COLOR_CALIBRATION_POLYGON = (255, 255, 255)
COLOR_LEGEND_TEXT = (255, 255, 255)
COLOR_LEGEND_BG = (100, 100, 100)

# --- NEW: Shop System & Items ---
SHOP_ITEMS = {
    # Highlights
    "hl_default": {"name": "Default Highlight", "price": 0, "type": "highlight", "fill": (0, 180, 180),
                   "border": (0, 255, 255)},
    "hl_fire": {"name": "Fire Highlight", "price": 5, "type": "highlight", "fill": (0, 69, 255),
                "border": (0, 165, 255)},
    "hl_ocean": {"name": "Ocean Highlight", "price": 5, "type": "highlight", "fill": (255, 102, 0),
                 "border": (255, 204, 0)},
    # Backgrounds
    "bg_default": {"name": "Default Background", "price": 0, "type": "background", "path": None},
    "bg_space": {"name": "Space Background", "price": 10, "type": "background",
                 "path": os.path.join(BACKGROUNDS_DIR, "bg_space.jpg")},
    "bg_gradient": {"name": "Gradient Background", "price": 10, "type": "background",
                    "path": os.path.join(BACKGROUNDS_DIR, "bg_gradient.jpg")},
}


# --- 2. State Management Enums ---
class GameState(Enum):
    MODE_SELECTION = 0
    MAP_DETECTION = 1
    QUIZ_ACTIVE = 2
    GAME_OVER = 3
    SHOP = 4


# --- 3. Question Data (Removed "Antarctica" question) ---
CONTINENT_QUESTIONS = [
    ["Point to Africa", "Africa"],
    ["Point to the continent with the Amazon Rainforest", "South-America"],
    ["Point to Europe", "Europe"],
    ["Point to the 'Land Down Under'", "Australia"],
    ["Point to North America", "North-America"],
    ["Point to the most populous continent", "Asia"]
]
COUNTRY_QUESTIONS = [
    ["Point to The United States", "USA"],
    ["Point to the country famous for samba", "Brazil"],
    ["Point to China", "China"],
    ["Point to India", "India"],
    ["Point to the largest country by area", "Russia"],
    ["Point to the country with the maple leaf flag", "Canada"],
    ["Point to the country with the Great Pyramids", "Egypt"],
    ["Point to the 'Land of the Rising Sun'", "Japan"]
]


# --- NEW: Player Data Functions ---
def load_player_data():
    try:
        with open(PLAYER_DATA_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "total_coins": 0,
            "unlocked_items": ["hl_default", "bg_default"],
            "equipped_items": {"highlight": "hl_default", "background": "bg_default"}
        }


def save_player_data(data):
    with open(PLAYER_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=4)


# --- Utility and Drawing Functions ---
def apply_post_processing(img: np.ndarray) -> np.ndarray:
    if not ENABLE_POST_PROCESSING:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=CONTRAST_CLIP_LIMIT, tileGridSize=(8, 8))
    v_enhanced = clahe.apply(v)
    img_contrast = cv2.cvtColor(cv2.merge([h, s, v_enhanced]), cv2.COLOR_HSV2BGR)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 4 + SHARPEN_INTENSITY, -1], [0, -1, 0]])
    return cv2.filter2D(src=img_contrast, ddepth=-1, kernel=sharpen_kernel)


def find_map_corners(img: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 30, 80)
    _, con_found = cvzone.findContours(img, img_canny, filter=[4])
    if con_found:
        approx = cv2.approxPolyDP(con_found[0]['cnt'], 0.02 * cv2.arcLength(con_found[0]['cnt'], True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return np.array([])


def get_warp_matrix(points: np.ndarray, size: tuple = (WIDTH, HEIGHT), flipped: bool = False) -> tuple | None:
    if points.shape[0] != 4:
        return None
    points_sorted = sorted(points, key=lambda x: x[1])
    if points_sorted[0][0] > points_sorted[1][0]:
        points_sorted[0], points_sorted[1] = points_sorted[1], points_sorted[0]
    if points_sorted[2][0] > points_sorted[3][0]:
        points_sorted[2], points_sorted[3] = points_sorted[3], points_sorted[2]
    pts1 = np.float32(points_sorted)
    pts2 = np.float32([[size[0], size[1]], [0, size[1]], [size[0], 0], [0, 0]]) if flipped else np.float32(
        [[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    try:
        return cv2.getPerspectiveTransform(pts1, pts2), cv2.getPerspectiveTransform(pts2, pts1)
    except (cv2.error, np.linalg.LinAlgError):
        return None


def get_finger_location(img: np.ndarray, matrix: np.ndarray, detector: HandDetector) -> tuple | None:
    hands, _ = detector.findHands(img, draw=False, flipType=True)
    if not hands:
        return None
    point = hands[0]["lmList"][8][0:2]
    point_h = np.array([point[0], point[1], 1], dtype=np.float32)
    point_t_h = matrix @ point_h
    w = point_t_h[2]
    return (int(point_t_h[0] / w), int(point_t_h[1] / w)) if w != 0 else None


# --- MODIFIED: Update Selection Function with Sparkle Effect on Hover ---
def update_selection_and_draw_polygons(polygons: list, warped_point: tuple, img_to_draw_on: np.ndarray,
                                       entry_times: dict, hl_color: tuple) -> tuple:
    selected_country, touching_polygons = None, set()
    for i, (polygon, name) in enumerate(polygons):
        poly_id = f"{name}_{i}"
        poly_np = np.array(polygon, np.int32)
        is_touching = cv2.pointPolygonTest(poly_np, warped_point, False) >= 0 if warped_point else False
        if is_touching:
            touching_polygons.add(poly_id)
            if poly_id not in entry_times:
                entry_times[poly_id] = time.time()
            time_in_poly = time.time() - entry_times[poly_id]
            # If selection delay reached, display the country name.
            if time_in_poly >= SELECTION_DELAY_SECONDS:
                selected_country = name
                cvzone.putTextRect(img_to_draw_on, name, tuple(poly_np[0]), FONT_SCALE_COUNTRY_NAME, 2,
                                   font=FONT_STYLE, colorR=hl_color, offset=10)
                current_color = hl_color
            else:
                current_color = hl_color
                angle = int((time_in_poly / SELECTION_DELAY_SECONDS) * 360)
                cv2.ellipse(img_to_draw_on, warped_point, (40, 40), -90, 0, angle, hl_color, 10)
            # ---------- Sparkle effect on the country currently highlighted ----------
            centroid = np.mean(poly_np, axis=0).astype(int)
            num_sparkles = 5
            for s in range(num_sparkles):
                angle_s = time.time() * 10 + s * (2 * math.pi / num_sparkles)
                # Oscillate the radius a bit for animation.
                radius = 20 + 5 * math.sin(time.time() * 3 + s)
                spark_x = int(centroid[0] + radius * math.cos(angle_s))
                spark_y = int(centroid[1] + radius * math.sin(angle_s))
                cv2.circle(img_to_draw_on, (spark_x, spark_y), 3, (255, 255, 255), -1)
            # --------------------------------------------------------------------------
            cv2.polylines(img_to_draw_on, [poly_np], True, current_color, 4)
    for poly_id in list(entry_times.keys()):
        if poly_id not in touching_polygons:
            del entry_times[poly_id]
    return img_to_draw_on, selected_country


def draw_hud(img: np.ndarray, game_state: dict):
    q, idx, score = game_state['questions'], game_state['current_question'], game_state['total_score']
    if idx < len(q):
        cvzone.putTextRect(img, f"Q: {q[idx][0]}", (50, 50), FONT_SCALE_HUD_QUESTION, 2,
                           font=FONT_STYLE, colorR=COLOR_NEUTRAL, offset=10)
        cvzone.putTextRect(img, f"Score: {score}", (WIDTH - 300, 50), FONT_SCALE_HUD_SCORE, 2,
                           font=FONT_STYLE, colorR=COLOR_NEUTRAL, offset=20)
    feedback = game_state['feedback_text']
    if feedback:
        lines = feedback.split('\n')
        cvzone.putTextRect(img, lines[0], (WIDTH // 2 - 150, 50), FONT_SCALE_FEEDBACK_MAIN, 3,
                           font=FONT_STYLE, colorR=game_state['feedback_color'])
        if len(lines) > 1:
            cvzone.putTextRect(img, lines[1], (WIDTH // 2 - 200, 120), FONT_SCALE_FEEDBACK_SUB, 2,
                               font=FONT_STYLE, colorR=COLOR_NEUTRAL)
        game_state['feedback_counter'] += 1
        if game_state['feedback_counter'] >= FEEDBACK_DISPLAY_FRAMES:
            game_state.update({"feedback_text": None, "feedback_counter": 0, "highlight_correct_answer_name": None})
            if idx < len(q):
                game_state['current_question'] += 1


# --- Shop Drawing Function ---
def draw_shop(img: np.ndarray, player_data):
    img.fill(20)  # Dark gray background
    cvzone.putTextRect(img, "SHOP", (WIDTH // 2 - 100, 50), 3, 3, colorR=COLOR_NEUTRAL)
    cvzone.putTextRect(img, f"Your Coins: {player_data['total_coins']}", (WIDTH - 400, 60), 2, 2, colorR=COLOR_CORRECT)
    y_pos = 150
    item_keys = list(SHOP_ITEMS.keys())
    for i, item_id in enumerate(item_keys):
        item = SHOP_ITEMS[item_id]
        cvzone.putTextRect(img, f"[{i + 1}] {item['name']}", (100, y_pos), FONT_SCALE_SHOP, 2, colorR=COLOR_LEGEND_BG)
        if item['type'] == 'highlight':
            cv2.rectangle(img, (550, y_pos - 20), (610, y_pos + 20), item['border'], -1)
            cv2.rectangle(img, (555, y_pos - 15), (605, y_pos + 15), item['fill'], -1)
        status_text = ""
        color = COLOR_WRONG
        if item_id in player_data['unlocked_items']:
            if player_data['equipped_items'].get(item['type']) == item_id:
                status_text = "Equipped"
                color = COLOR_CORRECT
            else:
                status_text = "Equip"
                color = COLOR_NEUTRAL
        else:
            status_text = f"Buy ({item['price']} Coins)"
        cvzone.putTextRect(img, status_text, (WIDTH - 450, y_pos), FONT_SCALE_SHOP, 2, colorR=color)
        y_pos += 80
    return item_keys


def draw_legend(img: np.ndarray, app_state: GameState, country_mode_available: bool):
    # Use a smaller legend font scale on the main menu (mode selection)
    if app_state == GameState.MODE_SELECTION:
        scale = 1.2
    else:
        scale = FONT_SCALE_LEGEND

    if app_state == GameState.MODE_SELECTION:
        text = "[1] Continents  |  [2] Countries  |  [S] Shop  |  [Q] Quit"
    elif app_state == GameState.MAP_DETECTION:
        text = "[C] Confirm Map  |  [Q] Quit"
    elif app_state == GameState.QUIZ_ACTIVE:
        text = "[F] Flip Map  |  [R] Recalibrate  |  [Q] Quit"
    elif app_state == GameState.GAME_OVER:
        text = "[S] Shop  |  [R] New Game  |  [Q] Quit"
    elif app_state == GameState.SHOP:
        text = "[1-9] Buy/Equip  |  [B] Back to Menu"
    cvzone.putTextRect(img, text, (30, HEIGHT - 50), scale, 2, font=FONT_STYLE,
                       colorT=COLOR_LEGEND_TEXT, colorR=COLOR_LEGEND_BG, offset=10, border=1)


def main():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.6)
    try:
        pygame.mixer.init()
        correct_sound = pygame.mixer.Sound(CORRECT_SOUND_PATH)
        wrong_sound = pygame.mixer.Sound(WRONG_SOUND_PATH)
    except Exception as e:
        print(f"Warning: Sound disabled. Error: {e}")
        correct_sound, wrong_sound = None, None

    player_data = load_player_data()
    print("Player data loaded:", player_data)

    background_images = {}
    for item_id, item in SHOP_ITEMS.items():
        if item['type'] == 'background' and item['path']:
            try:
                background_images[item_id] = cv2.imread(item['path'])
            except Exception as e:
                print(f"Warning: Could not load background '{item['path']}'. Error: {e}")

    try:
        with open(CONTINENT_POLYGONS_PATH, 'rb') as f:
            continent_polygons = pickle.load(f)
    except FileNotFoundError:
        print(f"FATAL: '{CONTINENT_POLYGONS_PATH}' not found.")
        return
    try:
        with open(COUNTRY_POLYGONS_PATH, 'rb') as f:
            country_polygons = pickle.load(f)
            country_mode_available = True
    except FileNotFoundError:
        print(f"Info: '{COUNTRY_POLYGONS_PATH}' not found.")
        country_polygons, country_mode_available = [], False

    game_state_defaults = {"current_question": 0, "total_score": 0, "feedback_text": None,
                           "feedback_color": COLOR_NEUTRAL, "feedback_counter": 0, "country_entry_times": {},
                           "highlight_correct_answer_name": None, "score_saved": False}
    game_state = game_state_defaults.copy()
    polygons, game_state["questions"] = [], []

    def reset_game():
        game_state.update(game_state_defaults.copy())
        print("Game progress/score reset.")

    app_state = GameState.MODE_SELECTION
    confirmed_corners, warp_matrix, map_is_flipped = None, None, False
    shop_item_keys = []

    while True:
        success, img = cap.read()
        if not success:
            break
        img_output = img.copy()

        if app_state == GameState.SHOP:
            img_output = np.full((HEIGHT, WIDTH, 3), COLOR_BACKGROUND, np.uint8)
            shop_item_keys = draw_shop(img_output, player_data)

        elif app_state == GameState.MODE_SELECTION:
            img_output = np.full((HEIGHT, WIDTH, 3), COLOR_BACKGROUND, np.uint8)
            # Adjusted Y positions for the mode options
            cvzone.putTextRect(img_output, "Choose a Game Mode", (WIDTH // 2 - 300, 80), FONT_SCALE_MENU, 3)
            cvzone.putTextRect(img_output, "[1] Continents Quiz", (WIDTH // 2 - 250, 200), FONT_SCALE_MENU, 3,
                               colorR=COLOR_NEUTRAL)
            color = COLOR_NEUTRAL if country_mode_available else COLOR_LEGEND_BG
            cvzone.putTextRect(img_output, "[2] Countries Quiz", (WIDTH // 2 - 250, 280), FONT_SCALE_MENU, 3,
                               colorR=color)
            cvzone.putTextRect(img_output, "[S] Shop", (WIDTH // 2 - 250, 360), FONT_SCALE_MENU, 3,
                               colorR=COLOR_LEGEND_BG)

        elif app_state == GameState.MAP_DETECTION:
            detected_corners = find_map_corners(img)
            if detected_corners.size > 0:
                cv2.drawContours(img_output, [detected_corners.astype(int)], -1, COLOR_MAP_BORDER, 3)
                cvzone.putTextRect(img_output, "Map Detected! Press 'c' to confirm.", (50, 50), 2.0, 2,
                                   font=FONT_STYLE, colorR=COLOR_MAP_BORDER)
                if (matrices := get_warp_matrix(detected_corners)):
                    _, inv_warp_matrix = matrices
                    calib_overlay = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    for p, _ in polygons:
                        cv2.polylines(calib_overlay, [np.array(p, np.int32)], True, COLOR_CALIBRATION_POLYGON, 2)
                    img_output = cv2.add(img_output,
                                         cv2.warpPerspective(calib_overlay, inv_warp_matrix, (WIDTH, HEIGHT)))
            else:
                cvzone.putTextRect(img_output, "Searching for map...", (50, 50), 2.0, 2, font=FONT_STYLE,
                                   colorR=COLOR_WRONG)

        elif app_state in [GameState.QUIZ_ACTIVE, GameState.GAME_OVER]:
            img_warped = cv2.warpPerspective(img, warp_matrix, (WIDTH, HEIGHT))
            img_warped = apply_post_processing(img_warped)
            if game_state["highlight_correct_answer_name"]:
                # When an answer has been confirmed (correct or wrong), highlight with sparkle effect.
                equipped_hl_id = player_data['equipped_items']['highlight']
                hl_colors = SHOP_ITEMS.get(equipped_hl_id, SHOP_ITEMS['hl_default'])
                COLOR_HIGHLIGHT_FILL, COLOR_HIGHLIGHT_BORDER = hl_colors['fill'], hl_colors['border']
                for p, name in polygons:
                    if name == game_state["highlight_correct_answer_name"]:
                        alpha = 0.5 + (np.sin(time.time() * 5) * 0.2)
                        overlay = img_warped.copy()
                        cv2.fillPoly(overlay, [np.array(p, np.int32)], COLOR_HIGHLIGHT_FILL)
                        img_warped = cv2.addWeighted(overlay, alpha, img_warped, 1 - alpha, 0)
                        cv2.polylines(img_warped, [np.array(p, np.int32)], True, COLOR_HIGHLIGHT_BORDER, 4)
                        # Sparkle effect on the confirmed polygon.
                        centroid = np.mean(np.array(p), axis=0).astype(int)
                        num_sparkles = 5
                        for s in range(num_sparkles):
                            angle = time.time() * 10 + s * (2 * math.pi / num_sparkles)
                            radius = 20 + 5 * math.sin(time.time() * 3 + s)
                            spark_x = int(centroid[0] + radius * math.cos(angle))
                            spark_y = int(centroid[1] + radius * math.sin(angle))
                            cv2.circle(img_warped, (spark_x, spark_y), 3, (255, 255, 255), -1)
                        break
            if app_state == GameState.QUIZ_ACTIVE:
                # Use player's equipped highlight color for hovering.
                equipped_hl_id = player_data['equipped_items']['highlight']
                hl_colors = SHOP_ITEMS.get(equipped_hl_id, SHOP_ITEMS['hl_default'])
                hl_border = hl_colors["border"]
                warped_point = get_finger_location(img, warp_matrix, detector)
                if warped_point:
                    cv2.circle(img_warped, warped_point, 15, COLOR_FINGER_TIP, cv2.FILLED)
                if not game_state['feedback_text']:
                    img_warped, sel_country = update_selection_and_draw_polygons(polygons, warped_point, img_warped,
                                                                                 game_state['country_entry_times'],
                                                                                 hl_border)
                    if sel_country and game_state['current_question'] < len(game_state['questions']):
                        correct_answer = game_state['questions'][game_state['current_question']][1]
                        if sel_country == correct_answer:
                            if correct_sound:
                                correct_sound.play()
                            game_state.update({"feedback_text": "CORRECT", "feedback_color": COLOR_CORRECT,
                                               "total_score": game_state['total_score'] + 1})
                        else:
                            if wrong_sound:
                                wrong_sound.play()
                            game_state.update(
                                {"feedback_text": f"WRONG\nCorrect: {correct_answer}", "feedback_color": COLOR_WRONG,
                                 "highlight_correct_answer_name": correct_answer})
            equipped_bg_id = player_data['equipped_items']['background']
            bg_img = background_images.get(equipped_bg_id)
            if bg_img is not None:
                img_output = cv2.resize(bg_img, (WIDTH, HEIGHT))
            else:
                img_output = np.full((HEIGHT, WIDTH, 3), COLOR_BACKGROUND, np.uint8)
            pad_x, pad_y = int(WIDTH * PAD_RATIO), int(HEIGHT * PAD_RATIO)
            img_map_resized = cv2.resize(img_warped, (WIDTH - 2 * pad_x, HEIGHT - 2 * pad_y))
            img_output[pad_y:pad_y + HEIGHT - 2 * pad_y, pad_x:pad_x + WIDTH - 2 * pad_x] = img_map_resized
            if app_state == GameState.QUIZ_ACTIVE:
                draw_hud(img_output, game_state)
                if game_state['current_question'] >= len(game_state['questions']):
                    app_state = GameState.GAME_OVER
            elif app_state == GameState.GAME_OVER:
                score, total = game_state['total_score'], len(game_state['questions'])
                cvzone.putTextRect(img_output, "GAME COMPLETE!", (WIDTH // 2 - 300, HEIGHT // 2 - 100),
                                   FONT_SCALE_GAMEOVER_MAIN, 4, font=FONT_STYLE, offset=30, colorR=COLOR_CORRECT)
                cvzone.putTextRect(img_output, f"Final Score: {score}/{total}", (WIDTH // 2 - 250, HEIGHT // 2 + 10),
                                   FONT_SCALE_GAMEOVER_SCORE, 3, font=FONT_STYLE, offset=20, colorR=COLOR_NEUTRAL)
                if not game_state['score_saved']:
                    player_data['total_coins'] += score
                    save_player_data(player_data)
                    game_state['score_saved'] = True
                    print(f"Added {score} coins. New total: {player_data['total_coins']}")
        draw_legend(img_output, app_state, country_mode_available)
        cv2.imshow("Geography Quiz", img_output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if app_state == GameState.SHOP:
            if ord('1') <= key <= ord('9'):
                item_idx = key - ord('1')
                if item_idx < len(shop_item_keys):
                    item_id = shop_item_keys[item_idx]
                    item = SHOP_ITEMS[item_id]
                    if item_id not in player_data['unlocked_items']:
                        if player_data['total_coins'] >= item['price']:
                            player_data['total_coins'] -= item['price']
                            player_data['unlocked_items'].append(item_id)
                            print(f"Purchased {item['name']}!")
                            save_player_data(player_data)
                        else:
                            print("Not enough coins!")
                    else:
                        player_data['equipped_items'][item['type']] = item_id
                        print(f"Equipped {item['name']}!")
                        save_player_data(player_data)
            elif key == ord('b'):
                app_state = GameState.MODE_SELECTION

        elif key == ord('s') and app_state in [GameState.MODE_SELECTION, GameState.GAME_OVER]:
            app_state = GameState.SHOP

        elif app_state == GameState.MODE_SELECTION:
            mode_selected = False
            questions = []
            if key == ord('1'):
                polygons, questions = continent_polygons, CONTINENT_QUESTIONS
                mode_selected = True
            elif key == ord('2') and country_mode_available:
                polygons, questions = country_polygons, COUNTRY_QUESTIONS
                mode_selected = True
            if mode_selected:
                shuffled_q = questions.copy()
                random.shuffle(shuffled_q)
                game_state["questions"] = shuffled_q
                reset_game()
                app_state = GameState.MAP_DETECTION

        elif key == ord('c') and app_state == GameState.MAP_DETECTION:
            if (corners := find_map_corners(img)).size > 0 and (matrices := get_warp_matrix(corners)):
                confirmed_corners, map_is_flipped = corners.copy(), False
                warp_matrix, _ = matrices
                app_state = GameState.QUIZ_ACTIVE
                print("Map Confirmed.")

        elif key == ord('f') and app_state == GameState.QUIZ_ACTIVE and confirmed_corners is not None:
            map_is_flipped = not map_is_flipped
            if (matrices := get_warp_matrix(confirmed_corners, flipped=map_is_flipped)):
                warp_matrix, _ = matrices
                print(f"Map flipped: {map_is_flipped}")

        elif key == ord('r'):
            if app_state == GameState.GAME_OVER:
                app_state = GameState.MODE_SELECTION
                print("\n----- NEW GAME -----")
            else:
                app_state = GameState.MAP_DETECTION
                print("\n----- RECALIBRATING MAP -----")
            confirmed_corners, warp_matrix = None, None

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()