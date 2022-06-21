from warm_start import main
import streamlit as st

prediction_output = main("TeacherForcing")
st.title("Anime music generator with neural network")
st.markdown('this website is application for generates midi audio files with TeacherForcing and other models')
st.image('sample/nn.png')
st.header('Sample result of model generation')
audio_file = open('sample/sample.wav','rb') #enter the filename with filepath
audio_bytes = audio_file.read() #reading the file 
st.audio(audio_bytes, format='audio/wav')
st.header('Let generated your midi audio files with TeacherForcing and other models')


# set app title
app_mode = st.radio("Select models", ["Tobyfox",
                                          "BiThreefold", "Tutorial"])
if app_mode == "BiThreefold":
    st.header("what is BiThreefold model?")
    st.markdown("BiThreefold is kowrjoivkergkm3rojvowkvwmrogjowks,vpwm3gjonwkvwrmkon kspvmwpgi2jwvk3t4mgkmkovksokvopwkv lpfmwpvqew]ofko2gflppsmvpwkkpl opkwopkfopggm")
    st.header("music generation with BiThreefold")
    with st.spinner('Wait for it...'):
        main("BiLSTM", prediction_output)
    st.success("generrated midi with BiThreefold success")
#     midi_data = pretty_midi.PrettyMIDI('output/warm_start_BiLSTM.mid')
#     audio_data = midi_data.fluidsynth()
#     audio_data = np.int16(
#             audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
#         )
#     virtualfile = io.BytesIO()
#     wavfile.write(virtualfile, 44100, audio_data)
#     st.audio(virtualfile)
    st.download_button(
    label="DOWNLOAD MIDI!",
    data="trees",
    file_name="output/warm_start_BiLSTM.mid")
    
elif app_mode == "Tutorial":
    st.header("what is Tutorial model?")
    st.markdown("Tutorial model is kowrjoivkergkm3rojvowkvwmrogjowks,vpwm3gjonwkvwrmkon kspvmwpgi2jwvk3t4mgkmkovksokvopwkv lpfmwpvqew]ofko2gflppsmvpwkkpl opkwopkfopggm")
    st.header("music generation with Tutorial model")
    with st.spinner('Wait for it...'):
        main("Tutorial", prediction_output)
    st.success("generrated midi with Tutorial model success")
    st.download_button(
    label="DOWNLOAD MIDI!",
    data="trees",
    file_name="output/swarm_start_Tutorial.mid")
    
elif app_mode == "Tobyfox":
    st.header("what is Tobyfox model?")
    st.markdown("Tobyfox is kowrjoivkergkm3rojvowkvwmrogjowks,vpwm3gjonwkvwrmkon kspvmwpgi2jwvk3t4mgkmkovksokvopwkv lpfmwpvqew]ofko2gflppsmvpwkkpl opkwopkfopggm")
    st.header("music generation with Tobyfox")
    with st.spinner('Wait for it...'):
        main("SingleLSTM", prediction_output)
    st.success("generrated midi with Tobyfox success")
    st.download_button(
    label="DOWNLOAD MIDI!",
    data="trees",
    file_name="output/warm_start_SingleLSTM.mid")
