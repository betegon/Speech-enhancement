import librosa
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
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sr)

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
    """This function take an audio and split into several frame
       in a numpy matrix of size (nb_frame,frame_length)"""

    sequence_sample_length = sound_data.shape[0]

    sound_data_list = [sound_data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    sound_data_array = np.vstack(sound_data_list)

    return sound_data_array















audio_dir = 'spectrogramVisualizing/good'
clean = 'clean.wav'
mixed = 'mixed.wav'
noise = 'noise.wav'
sample_rate = 16000  # Hz
frame_length = sample_rate + 64  # a bit more than sample_rate for avoiding overlapping.
min_duration = 1  # Seconds
hop_length_frame = sample_rate + 64

y_clean, sr_clean = librosa.load(os.path.join(audio_dir, clean), sr=sample_rate)
y_mixed, sr_mixed = librosa.load(os.path.join(audio_dir, mixed), sr=sample_rate)
y_noise, sr_noise = librosa.load(os.path.join(audio_dir, noise), sr=sample_rate)
# Total duration = y/sr
total_duration_clean = librosa.get_duration(y=y_clean, sr=sr_clean)
total_duration_mixed = librosa.get_duration(y=y_mixed, sr=sr_mixed)
total_duration_noise = librosa.get_duration(y=y_noise, sr=sr_noise)
print("Duration clean: {} seconds.".format(total_duration_clean))
print("Duration mixed: {} seconds.".format(total_duration_mixed))
print("Duration noise: {} seconds.".format(total_duration_noise))

fig, axs = plt.subplots(4, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
fig.suptitle('Time series')
axs[0].plot(y_clean)
axs[0].set_title('Clean voice', loc='right')

axs[1].plot(y_mixed)
axs[1].set_title('Mixed voice', loc='right')
axs[2].plot(y_noise)
axs[2].set_title('Noise', loc='right')
axs[3].plot(y_mixed - y_noise)
axs[3].set_title('Mixed voice - Noise = Clean voice', loc='right')

plt.show()

audio_files_to_numpy(audio_dir, [clean, mixed, noise], sample_rate, frame_length, hop_length_frame, min_duration)
