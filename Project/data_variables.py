import os
import numpy as np
import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join("Dataset")

sign_actions = np.array(['Pozdrav', 'Hvala', 'Oprosti', 'Ja', 'Naocale', 'Kuca'])
num_of_sequences = 50
sequence_length = 60
