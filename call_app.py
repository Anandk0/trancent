import streamlit as st
import requests
import tempfile
import os
import time

AGENT_URL = "http://127.0.0.1:8002/chat"

st.set_page_config(
    page_title="Jain MBA Admissions",
    page_icon="��",
    layout="centered"
)

st.title("📞 Jain MBA Admissions")
st.caption("AI Admission Counselor")

# -----------------------------
# Session state
# -----------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = f"web-{int(time.time())}"

if "call_active" not in st.session_state:
    st.session_state.call_active = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Header
# -----------------------------

st.markdown(
    """
    <div style="
        padding:20px;
        border-radius:15px;
        background:#f5f5f5;
        text-align:center;
        margin-bottom:20px;
    ">
        <h2>Jain College of Engineering and Research</h2>
        <p>Regular MBA Admissions</p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Call controls
# -----------------------------

if not st.session_state.call_active:

    if st.button(
        "📞 Start Call",
        type="primary",
        use_container_width=True
    ):
        st.session_state.call_active = True
        st.rerun()

else:

    st.success("🟢 Call connected")

    if st.button(
        "🔴 End Call",
        use_container_width=True
    ):
        st.session_state.call_active = False
        st.rerun()


# -----------------------------
# Conversation history
# -----------------------------

if st.session_state.messages:

    st.subheader("Conversation")

    for message in st.session_state.messages:

        if message["role"] == "user":
            st.markdown(
                f"**You:** {message['text']}"
            )

        else:
            st.markdown(
                f"**Divya:** {message['text']}"
            )

            if message.get("audio"):
                st.audio(
                    message["audio"],
                    format="audio/wav"
                )


# -----------------------------
# Microphone
# -----------------------------

if st.session_state.call_active:

    st.divider()

    st.subheader("🎙️ Speak")

    audio = st.audio_input(
        "Press and speak"
    )

    if audio is not None:

        # Save browser audio temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(audio.getvalue())
            audio_path = tmp.name

        try:

            with st.spinner("Divya is listening..."):

                with open(audio_path, "rb") as audio_file:

                    response = requests.post(
                        AGENT_URL,
                        files={
                            "file": (
                                "audio.wav",
                                audio_file,
                                "audio/wav"
                            )
                        },
                        data={
                            "session_id":
                                st.session_state.session_id,
                            "language": "hi"
                        },
                        timeout=120
                    )

            if response.status_code != 200:

                st.error(
                    f"Agent API error: {response.status_code}"
                )

                st.code(response.text)

            else:

                result = response.json()

                if result.get("status") != "success":

                    st.error(
                        result.get(
                            "message",
                            "Agent failed"
                        )
                    )

                else:

                    transcript = result["transcript"]
                    reply = result["reply"]
                    tts_text = result["tts_text"]
                    audio_file_path = result["audio_file"]

                    # Store conversation
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "text": transcript
                        }
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "text": reply,
                            "audio": audio_file_path
                        }
                    )

                    st.rerun()

        except Exception as e:

            st.error(
                f"Could not connect to Agent API: {e}"
            )

        finally:

            if os.path.exists(audio_path):
                os.remove(audio_path)

