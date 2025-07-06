![alt text](thumbnail.gif "gif")

# Geography Quiz Game

A finger tracking geography quiz that lets users label regions on a world map and then play a game by pointing to continents/countries with their finger. Uses OpenCV, cvzone, and hand tracking for an interactive learning experience.

---

## Table of Contents

* [Features](#features)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)

  * [1. Defining Polygons (MapPoly.py)](#1-defining-polygons-mappolypy)
  * [2. Playing the Game (GeoGuess.py)](#2-playing-the-game-geoguesspy)
* [Controls & Keys](#controls--keys)
* [Configuration](#configuration)

---

## Features

* Automatic detection of map corners for perspective transformation
* Mouse driven polygon labeling of multiple disjoint regions under the same name (eg North-America consisting of Canada, Usa etc)
* Finger tracking via webcam to select regions by holding for a configurable duration
* Quiz questions: point to the specified continent or country
* Live score tracking and reset/skip functionality

---

## Repository Structure

```txt
├── MapPoly.py       # Script to define and save map polygons
├── GeoGuess.py      # Main quiz game script
├── darkmap.png      # Static map image used for labeling
├── countries.p      # Pickle file storing labeled polygons
├── requirements.txt
└── README.md        
```

---

## Requirements

* Python 3.7+
* OpenCV (`cv2`)
* NumPy
* cvzone
* pickle (standard library)

Install dependencies with:

```bash
pip install opencv-python numpy cvzone mediapipe
```

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/donsolo-khalifa/GeoGame.git


   cd GeoGame

   ```


---

2. Install the Python dependencies:
```bash
pip install -r requirements.txt
```

*(Alternatively, manually install OpenCV, NumPy, and cvzone as shown above.)*

---

## Usage

### 1. Defining Polygons (MapPoly.py)

This script allows you to click and label regions on the map:

```bash
python MapPoly.py
```

1. The window displays the map and highlights detected corners in green.
2. Press **c** to confirm the detected map and enter polygon labeling mode.
3. Click to add vertices of a polygon. After selecting at least 3 points:

   * Press **s** to save the polygon and assign a name (e.g., "North-America").
   * If labeling disjoint regions with the same name, repeat the process and use the same name.
4. Press **x** to undo the last saved polygon.
5. Press **d** to return to corner detection mode.
6. Press **q** to save all polygons to `countries.p` and exit.

### 2. Playing the Game (GeoGuess.py)

Once you have `countries.p`, run the quiz game:

```bash
python GeoGuess.py
```

1. The script detects map corners and then warps the live webcam feed to the map perspective.
2. Follow on screen instructions:

   * Use your index finger to point at the region corresponding to each question.
   * Hold for 2 seconds to select.
3. Score is updated in real time. After finishing all questions, press **r** to restart.

---

## Controls & Keys

| Key | Description                                                                  |
| --- | ---------------------------------------------------------------------------- |
| `c` | Confirm map detection and switch to polygon/game mode                        |
| `d` | Return to map corner detection mode                                          |
| `s` | (MapPoly) Save current polygon with a name; (GeoGuess) skip current question |
| `x` | Undo last saved polygon (MapPoly only)                                       |
| `q` | Quit and save polygons or exit game                                          |
| `r` | Restart quiz (GeoGuess only)                                                 |

---

## Configuration

* **Map Image**: Change `MAP_IMAGE_PATH` in `MapPoly.py` to use a different map.
* **Quiz Questions**: Edit the `questions` list in `GeoGuess.py` to customize prompts and answers.
* **Hold Duration**: Adjust `green_duration_threshold` in `create_overlay_image()` function in `GeoGuess.py` for selection dwell time.

---

