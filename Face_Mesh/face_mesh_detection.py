import cv2 as cv
import mediapipe as mp 
import time 

mp_face_mesh=mp.solutions.face_mesh
mp_drawing =mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pTime=0
cTime=0

cap= cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(  static_image_mode=False,max_num_faces=1,refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success,image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
             mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
             mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
             mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        cv.imshow('Face Mesh', cv.flip(image, 1))
        if cv.waitKey(5) & 0xFF == ord('q'):
         break
cap.release()

