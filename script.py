import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Set up video capture
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get screen dimensions
screen_w, screen_h = pyautogui.size()

while True:
    ret, image = cam.read()
    if not ret:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    window_h, window_w, _ = image.shape

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to find face landmarks
    result = face_mesh.process(rgb_image)
    landmarks = result.multi_face_landmarks

    if landmarks:
        face_landmarks = landmarks[0].landmark

        # Get the specific landmarks for controlling the mouse
        for id, landmark in enumerate(face_landmarks[474:478]):
            x = int(landmark.x * window_w)
            y = int(landmark.y * window_h)

            if id == 1:
                mouse_x = int(x * screen_w / window_w)
                mouse_y = int(y * screen_h / window_h)
                pyautogui.moveTo(mouse_x, mouse_y)

            cv2.circle(image, (x, y), 3, (255, 255, 255), -1)

        # Get the landmarks for detecting eye blinks
        left_eye = [face_landmarks[145], face_landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * window_w)
            y = int(landmark.y * window_h)
            cv2.circle(image, (x, y), 3, (255, 255, 255), -1)

        # Detect eye blink
        left_eye_dist = abs(left_eye[0].y - left_eye[1].y)
        if left_eye_dist < 0.01:
            pyautogui.click()
            pyautogui.sleep(2)
            print("Mouse Clicked")

    # Display the image
    cv2.imshow("Eye Controller", image)

    # Break the loop if 'p' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('p'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
