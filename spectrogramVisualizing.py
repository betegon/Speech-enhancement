from librosa.output import write_wav
from librosa import load, get_duration
import matplotlib.pyplot as plt
import numpy as np
import os


def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (nb_frame,frame_length)
    for a sliding window of size hop_length_frame"""

    list_sound_array = []
    list_audio_below_min_duration = []

    for file in list_audio_files:
        # open the audio file
        y, sr = load(os.path.join(audio_dir, file), sr=sample_rate)
        total_duration = get_duration(y=y, sr=sr)

        if (total_duration >= min_duration):
            list_sound_array.append(
                audio_to_audio_frame_stack(y, frame_length, hop_length_frame))
            print(list_sound_array)
        else:
            list_audio_below_min_duration.append(os.path.join(audio_dir, file))
            print("The following file {} is below the min duration".format(os.path.join(audio_dir, file)))

    if list_audio_below_min_duration:
        print("These are the files below the minimum duration ({} seconds): ".format(min_duration))
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


def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    """This function take an audio and split into several frames
       in a numpy matrix of size (nb_frame,frame_length)"""

    sequence_sample_length = sound_data.shape[0]

    sound_data_list = [sound_data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    sound_data_array = np.vstack(sound_data_list)

    return sound_data_array


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

    fig, axs = plt.subplots(len(time_series_list), sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    for idx, time_serie in enumerate(time_series_list):
        fig.suptitle('Time series')
        axs[idx].plot(time_serie)
        axs[idx].set_title(time_series_titles[idx], loc='right')
    plt.show()


audio_dir = 'spectrogramVisualizing/medium'
clean = 'clean.wav'
mixed = 'mixed.wav'
noise = 'noise.wav'
sample_rate = 16000  # Hz
frame_length = sample_rate + 64  # a bit more than sample_rate for avoiding overlapping.
min_duration = 1  # Seconds
hop_length_frame = sample_rate + 64

y_clean, sr_clean = load(os.path.join(audio_dir, clean), sr=sample_rate)
y_mixed, sr_mixed = load(os.path.join(audio_dir, mixed), sr=sample_rate)
y_noise, sr_noise = load(os.path.join(audio_dir, noise), sr=sample_rate)

save_audio(y_clean, sample_rate, 'mispe.wav')

# Total duration = y/sr
total_duration_clean = get_duration(y=y_clean, sr=sr_clean)
total_duration_mixed = get_duration(y=y_mixed, sr=sr_mixed)
total_duration_noise = get_duration(y=y_noise, sr=sr_noise)
print("Duration clean: {} seconds.".format(total_duration_clean))
print("Duration mixed: {} seconds.".format(total_duration_mixed))
print("Duration noise: {} seconds.".format(total_duration_noise))

time_series = [y_clean, y_mixed, y_noise, y_mixed - y_noise]
titles = ['Clean voice', 'Mixed voice', 'Noise', 'Mixed voice - Noise = Clean voice']
plot_time_series(time_series, titles)


audio_files_to_numpy(audio_dir, [clean, mixed, noise], sample_rate, frame_length, hop_length_frame, min_duration)
