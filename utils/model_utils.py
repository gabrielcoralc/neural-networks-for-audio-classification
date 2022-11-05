import tensorflow as tf
import numpy as np

from IPython import display

# Segmemntar audio
from scipy.io.wavfile import write,read
import noisereduce as nr
import sounddevice as sd
import soundfile

import matplotlib.pyplot as plt

 # Los archivos de audio, solo tienen un canal (mono), se descartan el eje “channels” del array con axis=-1.
def decode_audio(audio_binary:str)->np.ndarray:
    """ Decodifica archivos de audio codificados en WAV en tensores "float32",  
        normalizados en el rango [-1.0, 1.0], retorna Devuelve el audio "float32" 
        y una frecuencia de muestreo."""
    audio_binary=tf.io.read_file(audio_binary)
    audio, _ = tf.audio.decode_wav(contents=audio_binary, desired_channels = 1)
    return tf.squeeze(audio, axis=-1).numpy()

def get_label(file_path:str)->str:
    """ Retorna la etiqueta del archivo de audio a partir de el nombre del archivo. """
      # Nota: se utiliza indexación en vez de “tuple unpacking” para permitir su funcionamiento en una grafica de TensorFlow.
    return file_path.split('/')[1].upper()[0]

def get_waveform_and_label(file_path:str):
    """Retorna una tupla con el nombre del archivo, la información del audio
         y su etiqueta.  """
    label = get_label(file_path)
    waveform = decode_audio(file_path)
    return file_path,waveform,label

def get_spectrogram(waveform:np.ndarray,input_l:int)->np.ndarray:
    """Retorna un array con el espectrograma a partir de la 
    información de un archivo de audio."""
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = input_l
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
      [input_l] - tf.shape(waveform),
      dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram.numpy()

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
        # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def recortar_audio(x,n):
    inicio=np.where(x>n)[0][0]
    final=np.where(x>n)[0][-1]
    return x[inicio:final]

def play_audio(df_series):
    print('Label:', df_series.label)
    print('Waveform shape:', df_series.audio.shape)
    print('Audio playback')
    display.display(display.Audio(df_series.audio, rate=df_series.audio.shape[0]))

def predict_vocal(model,vars,file_name,record=True):
    """Se emplea el modelo ya entrenado para hacer la prediccion de una vocal
        , bien sea por un archivo de audio o una grabacion que se activa al momento 
        de llamar a la funcion."""
    if record==True:
        # Sampling frequency
        freq = 16000

        # Recording duration
        duration = 1

        # Start recorder with the given values 
        # of duration and sample frequency
        recording = sd.rec(int(duration * freq), 
                           samplerate=freq, channels=1)

        # Record audio for the given number of seconds
        sd.wait()

        # This will convert the NumPy array to an audio
        # file with the given sampling frequency
        write(file_name+".wav", freq, recording)
    rate, data = read(file_name+".wav")
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    soundfile.write(file_name+".wav", reduced_noise, rate, subtype='PCM_16')


    d_audio=decode_audio(file_name+".wav")  # Using tensor flow to decode audio and normalize the audio signal between -1 and 1
    r_audio=recortar_audio(d_audio,vars[0]) # Slicing the audio signal between and amplitude value
    s_audio=get_spectrogram(r_audio,vars[1]) # Getting the spectogram from the signal and setting a fixed value for the lenght of the audio

    etiquetas=["A","E","I","O","U"]
    pred=etiquetas[np.argmax(model.predict(np.expand_dims(s_audio, axis=0)),axis=1)[0]] # Model predict for 1 value
    image=plt.imread("imagenes/"+pred+".jpg") #
    fig,ax = plt.subplots(1)


    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.imshow(image)
    plt.show()