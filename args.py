import argparse

parser = argparse.ArgumentParser(description='Speech enhancement,data creation, training and prediction')

#mode to run the program (options: data creation, training or prediction)
parser.add_argument('--mode',default='prediction', type=str, choices=['data_creation', 'training', 'prediction'])
#folders where to find noise audios and clean voice audio to prepare training dataset (mode data_creation)
parser.add_argument('--noise_dir', default='/home/betegon/Desktop/DENOISER/example/Speech-enhancement/Train/noise/', type=str)

parser.add_argument('--voice_dir', default='/home/betegon/Desktop/DENOISER/example/Speech-enhancement/Train/clean_voice/', type=str)
#folders where to save spectrograms, time series and sounds for training / QC
parser.add_argument('--path_save_spectrogram', default='/home/betegon/Desktop/DENOISER/example/Speech-enhancement/Train/spectrogram/', type=str)

parser.add_argument('--path_save_time_serie', default='/home/betegon/Desktop/DENOISER/example/Speech-enhancement/Train/time_serie', type=str)

parser.add_argument('--path_save_sound', default='/home/betegon/Desktop/DENOISER/example/Speech-enhancement/Train/sound', type=str)
# How many frames to create in data_creation mode.
# Each window (will be converted to 2D spectrogram) and will be a sample for training.
# nb_samples is simply the number of windows used.
parser.add_argument('--nb_samples', default=1, type=int)
#Training from scratch or pre-trained weights
parser.add_argument('--training_from_scratch',default=True, type=bool)
#folder of saved weights
parser.add_argument('--weights_folder', default='./weights', type=str)
#Nb of epochs for training
parser.add_argument('--epochs', default=10, type=int)
#Batch size for training
parser.add_argument('--batch_size', default=20, type=int)
#Name of saved model to read
parser.add_argument('--name_model', default='model_unet', type=str)
#directory where read noisy sound to denoise (prediction mode)
parser.add_argument('--audio_dir_prediction', default='./demo_data/test', type=str)
#directory to save the denoise sound (prediction mode)
parser.add_argument('--dir_save_prediction', default='./demo_data/save_predictions/', type=str)
#Noisy sound file to denoise (prediction mode)
parser.add_argument('--audio_input_prediction', default=['sp26_babble_sn15.wav'], type=list)
#File name of sound output of denoise prediction
parser.add_argument('--audio_output_prediction', default='denoise_t2.wav', type=str)
# Sample rate chosen to read audio
parser.add_argument('--sample_rate', default=8000, type=int)
# Minimum duration of audio files to consider
parser.add_argument('--min_duration', default=1.0, type=float)
# Training data will be frame of slightly above 1 second
parser.add_argument('--frame_length', default=8064, type=int)
# hop length for clean voice files separation (no overlap). This is the SLIDING WINDOW !!!!! It could be called sliding_window_size
parser.add_argument('--hop_length_frame', default=8064, type=int)
# hop length for noise files to blend (noise is splitted into several windows)
parser.add_argument('--hop_length_frame_noise', default=5000, type=int)


# Info extracted from librosa.core.stft
'''Choosing n_fft and hop_length_fft to have squared spectrograms

n_fft:int > 0 [scalar]
length of the windowed signal after padding with zeros. The number of rows in the STFT matrix D is (1 + n_fft/2).
The default value, n_fft=2048 samples, corresponds to a physical duration of 93 milliseconds at a sample rate of 22050 Hz,
    i.e. the default sample rate in librosa. This value is well adapted for music signals.
However, in speech processing, the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz.
In any case, we recommend setting n_fft to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm.

hop_length:int > 0 [scalar]
number of audio samples between adjacent STFT columns.
Smaller values increase the number of columns in D without affecting the frequency resolution of the STFT.
If unspecified, defaults to win_length / 4 (see below).
'''
parser.add_argument('--n_fft', default=255, type=int)

parser.add_argument('--hop_length_fft', default=63, type=int)
