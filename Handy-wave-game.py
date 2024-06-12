import cv2
import mediapipe as mp
import pyautogui

# Initialize the hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with the hands model
    results = hands.process(frame)
    # Convert the frame back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # If there are any hands detected
    if results.multi_hand_landmarks:
        # Loop through each hand
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw the hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get the coordinates of the index finger tip
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            # Get the label of the hand (left or right)
            label = hand_label.classification[0].label
            # Check if the index finger tip is on the right half of the frame and the hand is right
            if x > 0.5 and label == 'Right':
                # Press 'd' key
                pyautogui.press('d')
                #pyautogui.press('SPACE')
                # Display a message on the frame
                cv2.putText(frame, "Turn Right", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Press 'd' key
                pyautogui.press('a')
                # Display a message on the frame
                cv2.putText(frame, "Turn Left", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Show the frame
    cv2.imshow('Hand Gesture', frame)
    # Exit if the user presses 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
