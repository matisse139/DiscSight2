import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from groq import Groq
import cv2
import mediapipe as mp
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import math

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)

def draw_pose(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    skeleton = image.copy()

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            skeleton,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    return skeleton, results

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1]-b[1],
        c[0]-b[0]
    ) - np.arctan2(
        a[1]-b[1],
        a[0]-b[0]
    )

    angle = abs(radians*180/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle

# --- Page Configuration ---
st.set_page_config(page_title="DiscSight | Ultimate Analytics", layout="wide")

# --- UI Sidebar ---
st.sidebar.title("🥏 DiscSight Control")
uploaded_file = st.sidebar.file_uploader("Upload Game Footage (MP4)", type=['mp4'])
analysis_mode = st.sidebar.selectbox("Analyze Mode", ["Individual Form", "Team Possession", "Strategic Heatmaps"])
st.sidebar.divider()
st.sidebar.subheader("Training Parameters")
tracking_sensitivity = st.sidebar.slider("Trajectory Smoothing", 0.1, 1.0, 0.5)

# --- Main Dashboard ---
st.title("DiscSight: Ultimate Frisbee Performance Analytics")
st.markdown("Transforming observational coaching into actionable, data-driven insights.")

# --- Visual Analysis ---
c1, c2 = st.columns([0.65, 0.35])

with c1:
    st.subheader("Disc Trajectory Mapping")
    if uploaded_file:
        st.video(uploaded_file)
    else:
        st.info("Upload game footage to see YOLOv8 trajectory mapping overlay.")
        # Simulated Trajectory Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 10, 20, 30, 40], y=[0, 5, 8, 5, 0], mode='lines+markers', name="Disc Path"))
        fig.update_layout(title="Flight Path Visualization", height=400)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Throwing Mechanics Score")
    # Simulated form analysis scores
    form_data = pd.DataFrame({
        'Metric': ['Pivot Stability', 'Release Angle', 'Follow-through', 'Body Rotation'],
        'Score': [88, 92, 75, 85]
    })
    fig_radar = px.line_polar(form_data, r='Score', theta='Metric', line_close=True)
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)

# --- Strategic & Team Insights ---
st.divider()
tab1, tab2, tab3, tab4 = st.tabs([
    "🥏 Pro Comparison",
    "📋 Progress Tracker",
    "📊 Mechanics Breakdown",
    "🤖 AI Coach"
])

with tab1:

    st.subheader("Professional Player Comparison")

    reference_image = st.file_uploader(
        "Upload Ideal Player Image",
        type=["jpg","jpeg","png"],
        key="tobe"
    )

    user_image = st.file_uploader(
        "Upload Your Throw",
        type=["jpg","jpeg","png"],
        key="user"
    )

    if reference_image and user_image:

        ref = cv2.imdecode(
            np.frombuffer(reference_image.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        usr = cv2.imdecode(
            np.frombuffer(user_image.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        ref_pose, ref_results = draw_pose(ref)
        usr_pose, usr_results = draw_pose(usr)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Professional")
            st.image(
                cv2.cvtColor(ref_pose, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with col2:
            st.markdown("### Your Throw")

            overlay = usr.copy()

            if usr_results.pose_landmarks:

                mp_draw.draw_landmarks(
                    overlay,
                    usr_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        if (
            ref_results.pose_landmarks
            and
            usr_results.pose_landmarks
        ):

            ref_landmarks = ref_results.pose_landmarks.landmark
            usr_landmarks = usr_results.pose_landmarks.landmark

            ref_elbow = calculate_angle(
                (
                    ref_landmarks[11].x,
                    ref_landmarks[11].y
                ),
                (
                    ref_landmarks[13].x,
                    ref_landmarks[13].y
                ),
                (
                    ref_landmarks[15].x,
                    ref_landmarks[15].y
                )
            )

            usr_elbow = calculate_angle(
                (
                    usr_landmarks[11].x,
                    usr_landmarks[11].y
                ),
                (
                    usr_landmarks[13].x,
                    usr_landmarks[13].y
                ),
                (
                    usr_landmarks[15].x,
                    usr_landmarks[15].y
                )
            )

            diff = abs(ref_elbow - usr_elbow)

            similarity = max(
                0,
                100 - diff
            )

            st.metric(
                "Similarity Score",
                f"{similarity:.1f}%"
            )

            summary = f"""
You are an elite Ultimate Frisbee throwing coach.

Compare the user's throwing form to Tobe Decraene.

Reference elbow angle:
{ref_elbow:.1f}

User elbow angle:
{usr_elbow:.1f}

Difference:
{diff:.1f}

Give:

- Three differences
- Three coaching tips
- Keep under 150 words.
"""

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role":"system",
                        "content":"You are an Ultimate Frisbee biomechanics coach."
                    },
                    {
                        "role":"user",
                        "content":summary
                    }
                ]
            )

            st.markdown("## AI Coaching Summary")

            st.write(
                completion.choices[0].message.content
            )
    
 

with tab2:
    st.subheader("Personal Improvement Over Time")
    possession_data = pd.DataFrame({'Team': ['Offense', 'Defense'], 'Time': [65, 35]})
    st.bar_chart(possession_data.set_index('Team'))

with tab3:
    st.subheader("Full Mechanics Breakdown")
    st.dataframe(pd.DataFrame({
        'Mechanic': ['Hip Angle', 'Arm Path', 'Angle'],
        'Score': [85, 50, 96],
        'Improvement From Last': [+12, -5, 0]
    }))

with tab4:
    st.subheader("DiscSight AI Coach")

    st.write(
        "Ask questions about Ultimate Frisbee strategy, throwing mechanics, "
        "training, or the dashboard results."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm your DiscSight AI Coach. "
                    "Ask me anything about Ultimate Frisbee."
                ),
            }
        ]

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask the AI Coach...")

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            response_placeholder = st.empty()

            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are DiscSight AI Coach, an expert in "
                                "Ultimate Frisbee coaching, throwing mechanics, "
                                "strategy, offensive systems, defensive systems, "
                                "training, and sports analytics."
                            ),
                        }
                    ]
                    + st.session_state.messages,
                    temperature=0.5,
                    max_tokens=700,
                )

                answer = completion.choices[0].message.content

            except Exception as e:
                answer = f"Error: {e}"

            response_placeholder.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

# --- Export/Action ---
st.divider()
st.subheader("Coaching Action Plan")
if st.button("Generate Coaching Summary Report"):
    st.success("Report generated: Focus on 'Follow-through' mechanics for player #7.")
    st.download_button("Download PDF", data="Coaching Data Placeholder", file_name="coaching_report.pdf")

# --- Footer ---
st.divider()
st.caption("DiscSight AI | Bridging the gap in sports technology | Built with Computer Vision & YOLOv8")