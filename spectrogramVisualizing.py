from librosa.output import write_wav
from librosa import load, get_duration
import matplotlib.pyplot as plt
import numpy as np
import os

# TODO: Document That I have used zero padding for STFT
# TODO: Change values in args.py to get 16064Hz as frame rate  and all the other parameters.
# TODO: Rename audio_to_audio_frame_stack() to time_series_to_frame_stack()
# TODO: Substitute the old functions by the ones here.
# TODO: Handle different encoding types of audio + channels (2 vs 1 channel).
#       Best way is maybe convert from 2 to 1 channel.


def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (nb_frame,frame_length)
    for a sliding window of size hop_length_frame

    args:
        audio_dir (str): Directory where audios are located
        list_audio_files (list): string list of names of audio files.
        sample_rate (int): Sample rate of audios.
        frame_length (int): Length of frames.
        min_duration (int): Mininum duration of the audios to be added.
    """

    list_sound_array = []
    list_audio_below_min_duration = []

    for file in list_audio_files:
        # open the audio file
        y, sr = load(os.path.join(audio_dir, file), sr=sample_rate)
        total_duration = get_duration(y=y, sr=sr)

        if (total_duration >= min_duration):
            list_sound_array.append(
                audio_to_audio_frame_stack(
                    sound_data=y,
                    frame_length=frame_length,
                    hop_length_frame=hop_length_frame))
            print(list_sound_array)
        else:
            list_audio_below_min_duration.append(os.path.join(audio_dir, file))
            print("The following file {} is below the min duration".format(os.path.join(audio_dir, file)))

    if list_audio_below_min_duration:
        print("Following files are below minimum duration ({} seconds) and won't be included: ".format(min_duration))
        for audio_below_duration in list_audio_below_min_duration:
            print("  {}".format(audio_below_duration))

    if list_sound_array:
        print("Lista de arrays: \n\n")
        print(list_sound_array)
        print("\n\n\n\nLista de arrays ESTAKADA: \n\n")
        print(np.vstack(list_sound_array))
        return np.vstack(list_sound_array)
    else:
        print("There aren't any files above minimum duration ({} seconds).".format(min_duration))
        return


def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame, include_if_bigger_than=0.2):
    """This function take an audio and split into several frames
       in a numpy matrix of size (nb_frame,frame_length)

    args:
        sound_data (list): List of amplitudes returned from librosa.load().
        frame_length (int): Length of frames.
        hop_length_frame (int): Sliding window.
        include_if_bigger_than (float): Value between 0 and 1. Default to 0.2. Include last window (that will be padded) if it is greater than a percentage of the sliding window.

    note: to match window size (i.e hop_length_frame), it applies a zero padding.

    """

    sound_data_list = []
    time_series_length = sound_data.shape[0]
    for start in range(0, time_series_length, hop_length_frame):
        frame = sound_data[start:(start + frame_length)]
        if(frame.shape[0] == hop_length_frame):
            sound_data_list.append(frame)
        elif(frame.shape[0] < hop_length_frame and frame.shape[0] > (include_if_bigger_than * hop_length_frame)):
            # if it is the last element, add zero padding to match hop_length_frame
            frame = np.pad(frame, (0, hop_length_frame-frame.shape[0]), 'constant')
            sound_data_list.append(frame)

    return np.vstack(sound_data_list)


def save_audio(y, sample_rate, output_name='audio_ouput.wav'):
    '''
    Save audio file given y (amplitude values) and sample_rate.
    By default, output name is 'audio_output.wav'
    '''
    write_wav(output_name, y_mixed - y_noise, sample_rate)


def plot_time_series(time_series_list, time_series_titles):
    '''Plot a list of time series in different subfigures.
    args:
      time_series_list:   List of time_series to plot
      time_series_titles: list of strings containing titles of subplots
    '''
    if (len(time_series_list) != len(time_series_titles)):
        raise Exception("time_series_list and time_series_titles should have the same lenght. There should be a title for each time serie.")
    fig, axs = plt.subplots(len(time_series_list), sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    for idx, time_serie in enumerate(time_series_list):
        fig.suptitle('Time series')
        axs[idx].plot(time_serie)
        axs[idx].set_title(time_series_titles[idx], loc='right')
    plt.show()


audio_dir = 'spectrogramVisualizing/medium'
clean_filename = 'clean.wav'
mixed_filename = 'mixed.wav'
noise_filename = 'noise.wav'
sample_rate = 16000  # Hz
frame_length = sample_rate + 64  # a bit more than sample_rate for avoiding overlapping.
min_duration = 1  # Seconds
hop_length_frame = sample_rate + 64

y_clean, sr_clean = load(os.path.join(audio_dir, clean_filename), sr=sample_rate)
y_mixed, sr_mixed = load(os.path.join(audio_dir, mixed_filename), sr=sample_rate)
y_noise, sr_noise = load(os.path.join(audio_dir, noise_filename), sr=sample_rate)

# Total duration = y/sr
total_duration_clean = get_duration(y=y_clean, sr=sr_clean)
total_duration_mixed = get_duration(y=y_mixed, sr=sr_mixed)
total_duration_noise = get_duration(y=y_noise, sr=sr_noise)
print("Duration clean: {} seconds.".format(total_duration_clean))
print("Duration mixed: {} seconds.".format(total_duration_mixed))
print("Duration noise: {} seconds.".format(total_duration_noise))

# PLOT TIME SERIES
time_series = [y_clean, y_mixed, y_noise, y_mixed - y_noise]
titles = ['Clean voice', 'Mixed voice', 'Noise', 'Mixed voice - Noise = Clean voice']
plot_time_series(time_series, titles)

''''STEPS FOR CREATING DATASET
1. Audio files to numpy (audio_files())
    1.1. Load .wav
    1.2. audio to audio frame stack (audio_to_audio_frame_stack())
    1.3. Append result from 1.2 to audio list (list_sound_array)
2. Blend noise randomly (blend_noise_randomly())
3. noisy_voice_long = reshape
4. save noisy_voice_long
5. Repeat 3. & 4. for voice_long and noise_long
'''

# 1. Audio files to numpy
audio_files_to_numpy(audio_dir, [clean_filename, mixed_filename, noise_filename], sample_rate, frame_length, hop_length_frame, min_duration)

# 1.2 Audio to audio frame stack
list_sound_array = []
appending = audio_to_audio_frame_stack(y_clean, frame_length, hop_length_frame)
print("\ny_clean \nappending.\n")
print(y_clean)
print(appending)
print(y_clean.shape)
if appending.any():
    print(appending.shape)
# 1.3 Append result from 1.2 to list_sound_array
list_sound_array.append(appending)

























# Example of saving audio
save_audio(y_clean, sample_rate, 'mispe.wav')
