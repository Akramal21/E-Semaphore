import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
        text-align: center;
        
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Semaphore Detection')
st.sidebar.image('semaphore.png', width=320)
#st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Pilih Mode Aplikasi', ['Info','Run on Video'])

if app_mode =='Info':
    st.markdown("<h1 style='text-align: center; color: black;'> Sejarah Semaphore</h1>", unsafe_allow_html=True)
    st.markdown('---')
    #st.markdown("<hr/>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(' ')
    with c2:
        st.image("Chappe.jpg")
    with c3:
        st.write(' ')
    st.markdown("<h5 style='text-align: center; color: black;'>"
                "Dikutip dari arsip Canadian War Museum, teknik semafor tercipta di Perancis oleh Claude Chappe pada 1790 bersama saudara laki-lakinya yang bernama Abraham. "
                "Saat itu bertepatan dengan Revolusi Perancis. "
                "Chappe bersaudara mencari cara bagaimana menyampaikan pesan rahasia."
                "</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: black;'>"
                "Chappe bersaudara memanfaatkan serangkaian menara sebagai media untuk menyampaikan pesan. "
                "Di masing-masing menara itu dipasang semacam lengan dari kayu dan ditempatkan satu orang sebagai operator."
                "</h5>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.write(' ')
    with b2:
        st.image("2.jpg")
    with b3:
        st.write(' ')
    st.markdown("<h5 style='text-align: center; color: black;'>"
                "Akhirnya Semaphore digunakan juga di dalam Organisasi Pramuka. "
                "Masih dengan fungsi yang sama yaitu menyampaikan pesan dengan kode-kode gerakan tertentu."
                "</h5>", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    [data-testid="stTitle"][aria-expanded="true"] > div:first-child {
        width: 400px;
        text-align: center;
    }
    [data-testid="stTitle"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

elif app_mode =='Run on Video':
    st.markdown("<h1 style='text-align: center; color: black;'>Semaphore Detection</h1>", unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')
    st.image('1.jpg')
    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            with open('semaphore.pkl', 'rb') as f:
                model = pickle.load(f)

            vid = cv2.VideoCapture(0)

            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(vid.get(cv2.CAP_PROP_FPS))

            st.sidebar.text('Input Video')
            st.sidebar.video(tfflie.name)
            fps = 0
            i = 0
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            stop = st.button('Stop Webcam')

            with kpi1:
                st.markdown("**FrameRate**")
                kpi1_text = st.markdown("0")

            with kpi2:
                st.markdown("**Semaphore Terdeteksi**")
                kpi2_text = st.markdown("Belum Terdeteksi")

            with kpi3:
                st.markdown("**Probability**")
                kpi3_text = st.markdown("0")

            with kpi4:
                st.markdown("**Image Width**")
                kpi4_text = st.markdown("0")

            st.markdown("<hr/>", unsafe_allow_html=True)


            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                prevTime = 0

                while vid.isOpened():
                    i += 1
                    ret, frame = vid.read()
                    if not ret:
                        continue

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False

                    results = holistic.process(frame)

                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # 1. Tangan Kanan
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                              )

                    # 2. Tangan Kiri
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                              )

                    # 3. Deteksi Pose
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )

                    # Export coordinates
                    try:
                        # Extract Pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array(
                            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                        # Concate rows
                        row = pose_row
                        # Make Detections
                        X = pd.DataFrame([row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        print(body_language_class, body_language_prob)

                        # Grab ear coords
                        coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640, 480]).astype(int))

                        cv2.rectangle(frame,
                                      (coords[0], coords[1] + 5),
                                      (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                                      (245, 117, 16), -1)
                        cv2.putText(frame, body_language_class, coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        kpi2_text.write(
                            f"<h1 style='text-align: center; color: red;'>{body_language_class.split(' ')[0]}</h1>",
                            unsafe_allow_html=True)
                        kpi3_text.write(
                            f"<h1 style='text-align: center; color: red;'>{body_language_prob[np.argmax(body_language_prob)]}</h1>",
                            unsafe_allow_html=True)

                    except:
                        pass

                    currTime = time.time()
                    fps = 1 / (currTime - prevTime)
                    prevTime = currTime

                    # Dashboard
                    kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>",unsafe_allow_html=True)

                    kpi4_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)


                    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                    frame = image_resize(image=frame, width=720)
                    stframe.image(frame, channels='BGR', use_column_width=True)

            vid.release()
            st.markdown(' ## Contoh Kode Semaphore')
        else:
            st.markdown('Gunakan WebCam')


    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as pose:
            prevTime = 0

            while vid.isOpened():
                i +=1
                ret, frame = vid.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                #Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                #kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 720)
                stframe.image(frame,channels = 'BGR',use_column_width=True)

            vid.release()
            #out. release()