# This code is taken and slightly modified from
# https://github.com/openpifpaf/openpifpaf_posetrack/
# which is owned by Sven Kreiss and licensed under the MIT license
# For further information visit the link above

import numpy as np



KEYPOINTS = [
    'head_bottom',
    'head_top',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

SIGMAS = [
    0.08,  # 2 head_bottom ==> changed versus COCO
    0.06,  # 3 head_top ==> changed versus COCO
    0.079,  # 4 shoulders
    0.079,  # 5 shoulders
    0.072,  # 6 elbows
    0.072,  # 7 elbows
    0.062,  # 8 wrists
    0.062,  # 9 wrists
    0.107,  # 10 hips
    0.107,  # 11 hips
    0.087,  # 12 knees
    0.087,  # 13 knees
    0.089,  # 14 ankles
    0.089,  # 15 ankles
]

UPRIGHT_POSE = np.array([
    [-0.05, 9.0, 2.0],  # 'head_bottom',        # 2
    [0.05, 10.0, 2.0],  # 'head_top',       # 3
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 4
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 5
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 6
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 7
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 8
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 9
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 10
    [1.26, 4.0, 2.0],  # 'right_hip',       # 11
    [-1.4, 2.0, 2.0],  # 'left_knee',       # 12
    [1.4, 2.1, 2.0],  # 'right_knee',      # 13
    [-1.4, 0.0, 2.0],  # 'left_ankle',      # 14
    [1.4, 0.1, 2.0],  # 'right_ankle',     # 15
])

SKELETON = [
    [13, 11],
    [11, 9],
    [14, 12],
    [12, 10],
    [9, 10],
    [3, 9],
    [4, 10],
    [3, 5],
    [4, 6],
    [5, 7],
    [6, 8],
    [1, 3],
    [1, 4],
    [1, 2],
]

HFLIP = {
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}

CATEGORIES = ['person',]

SCORE_WEIGHTS = [3.0] * 3 + [1.0] * (len(KEYPOINTS) - 3)

DENSER_CONNECTIONS = [
    [4, 5],  # shoulders
    [6, 7],  # elbows
    [8, 9],  # wrists
    [12, 13],  # knees
    [14, 15],  # ankles
    [4, 8],  # shoulder - wrist
    [5, 9],
    [8, 10],  # wrists - hips
    [9, 11],
    [2, 8],  # headbottom - wrists
    [2, 9],
    [10, 13],  # hip knee cross
    [11, 12],
    [12, 15],  # knee ankle cross
    [13, 14],
    [4, 11],  # shoulders hip cross
    [5, 10],
    [4, 3],  # shoulders head top
    [5, 3],
    [4, 1],  # shoulders head nose
    [5, 1],
    [6, 2],  # elbows head_bottom
    [7, 2],  # elbows head_bottom
]

