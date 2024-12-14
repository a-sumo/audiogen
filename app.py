import streamlit as st
import torch
import soundfile as sf
from diffusers import StableAudioPipeline
from huggingface_hub import login
import os
import io

# Login to Hugging Face
login(token="your_hugging_face_token_here")

# Define constants
MODEL_NAME = "stabilityai/stable-audio-open-1.0"
LOCAL_MODEL_PATH = "./stable_audio_model"

def load_or_download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("Downloading model..."):
            pipe = StableAudioPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
            pipe.save_pretrained(LOCAL_MODEL_PATH)
        st.success("Model downloaded and saved locally.")
    else:
        with st.spinner("Loading model from local storage..."):
            pipe = StableAudioPipeline.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.float32)
        st.success("Model loaded from local storage.")

    return pipe.to("cpu")

@st.cache_resource
def get_model():
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    if not st.session_state.model_loaded:
        pipe = load_or_download_model()
        st.session_state.model_loaded = True
        return pipe
    else:
        return StableAudioPipeline.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.float32).to("cpu")

pipe = get_model()

def generate_audio(prompt, negative_prompt, num_inference_steps, audio_end_in_s, num_waveforms_per_prompt):
    # Set the seed for reproducibility
    generator = torch.Generator("cpu").manual_seed(0)

    # Generate audio
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        audio_end_in_s=audio_end_in_s,
        num_waveforms_per_prompt=num_waveforms_per_prompt,
        generator=generator,
    ).audios

    # Process the audio
    output = audio[0].T.float().cpu().numpy()

    # Save audio to a BytesIO object
    buffer = io.BytesIO()
    sf.write(buffer, output, pipe.vae.sampling_rate, format='wav')
    buffer.seek(0)

    return buffer.getvalue()

st.title("Stable Audio Generator")

with st.form("audio_generation"):
    prompt = st.text_input("Prompt", "an electronic music sample of kick.")
    negative_prompt = st.text_input("Negative Prompt", "Low quality.")
    num_inference_steps = st.slider("Number of inference steps", 50, 500, 200)
    audio_end_in_s = st.number_input("Audio duration (seconds)", min_value=1.0, max_value=60.0, value=10.0, step=0.1)
    num_waveforms_per_prompt = st.slider("Number of waveforms per prompt", 1, 10, 3)

    submitted = st.form_submit_button("Generate Audio")

if submitted:
    with st.spinner("Generating audio..."):
        try:
            audio_data = generate_audio(prompt, negative_prompt, num_inference_steps, audio_end_in_s, num_waveforms_per_prompt)
            
            # Create a download button
            st.download_button(
                label="Download generated audio",
                data=audio_data,
                file_name="generated_audio.wav",
                mime="audio/wav"
            )

            # Display an audio player
            st.audio(audio_data)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
