from progress.bar import IncrementalBar
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

# TODO: Fill Inference audio to match size for STFT
# TODO: Document That I have used zero padding for STFT
# TODO: Change values in args.py to get 16128Hz as frame rate  and all the other parameters.
# TODO: Rename audio_to_audio_frame_stack() to time_series_to_frame_stack()
# TODO: Substitute the old functions by the ones here.
# TODO: Start training from the weights provided in the repo.
# TODO: Padding noise with something different than zeros
#       (maybe with frames from earlier in the same noise audio).
#       So when the speakers stops talking, the noise persists.
# TODO: Think of a way to produce more samples by mixing noises or something.
#       Maybe increasing its amplitude. Check blend_noise_randomly()
# TODO: Handle different encoding types of audio + channels (2 vs 1 channel).
#       Best way is maybe convert from 2 to 1 channel.


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



def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (total_frames,frame_length)
    for a sliding window of size hop_length_frame

    Args:
        audio_dir (str): Directory where audios are located
        list_audio_files (list): string list of names of audio files.
        sample_rate (int): Sample rate of audios.
        frame_length (int): Length of frames.
        min_duration (int): Mininum duration of the audios to be added.

    Returns:
        np.ndarray: numpy ndarray shape (total_frames, frame_length).
    """

    list_sound_array = []
    list_audio_below_min_duration = []

    print("Total files to be processed: {}".format(len(list_audio_files)))

    for file in IncrementalBar('Audio file to numpy').iter(list_audio_files):
        # open the audio file
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sr)

        if (total_duration >= min_duration):
            list_sound_array.append(
                audio_to_audio_frame_stack(
                    sound_data=y,
                    frame_length=frame_length,
                    hop_length_frame=hop_length_frame))

        else:
            list_audio_below_min_duration.append(os.path.join(audio_dir, file))
            print("The following file {} is below the min duration".format(os.path.join(audio_dir, file)))

    if list_audio_below_min_duration:
        print("Following files are below minimum duration ({} seconds) and won't be included: ".format(min_duration))
        for audio_below_duration in list_audio_below_min_duration:
            print("  {}".format(audio_below_duration))

    if list_sound_array:
        return np.vstack(list_sound_array)
    else:
        print("There aren't any files above minimum duration ({} seconds).".format(min_duration))


def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame, include_if_bigger_than=0.2):
    """This function take an audio and split into several frames
       in a numpy matrix of size (number_of_frames,frame_length).

    Args:
        sound_data (list): List of amplitudes returned from librosa.load().
        frame_length (int): Length of frames.
        hop_length_frame (int): Sliding window.
        include_if_bigger_than (float): Value between 0 and 1. Default to 0.2. Include last window (that will be padded) if it is greater than a percentage of the sliding window.
    Returns:
        np.ndarray: Multidimensional array of shape (number_of_frames, frame_length).
                    I
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


def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
    """This function takes an audio and convert into spectrogram,
       it returns the magnitude in dB and the phase"""
    '''
    print("\naudio")
    print(audio)
    print(audio.shape)

    print("n_fft")
    print(n_fft)

    print("hop_length_fft")
    print(hop_length_fft)
    '''
    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)
    '''
    print("stftaudio")
    print(stftaudio)
    print(stftaudio.shape)
    '''
    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)
    '''
    print("stftaudio magnitude and phase shapes: ")
    print(stftaudio_magnitude_db.shape)
    print(stftaudio_phase.shape)
    '''
    return stftaudio_magnitude_db, stftaudio_phase


def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    """This function takes as input a numpy audio of size (nb_frame,frame_length), and return
    a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
    (nb_frame,dim_square_spec,dim_square_spec)"""
    nb_audio = numpy_audio.shape[0]
    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in IncrementalBar('Numpy audio to matrix spectrogram').iter(range(nb_audio)):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i])

    print('\nShape of Spectrograms generated: {}\n'.format(m_mag_db.shape[1:]))
    return m_mag_db, m_phase


def save_audio(y, sample_rate, output_name='audio_ouput.wav'):
    '''
    Save audio file given y (amplitude values) and sample_rate.
    By default, output name is 'audio_output.wav'

    Args:
        y (ndarray): shape should be (n,) or (2,n).
    '''
    librosa.output.write_wav(output_name, y, sample_rate)
    print("Duration of written audio is: {} seconds.".format(librosa.get_duration(y=y, sr=sample_rate)))


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




def main():
    audio_dir = 'spectrogramVisualizing/medium'
    sample_rate = 8000  # Hz
    frame_length = sample_rate + 64  # a bit more than sample_rate for avoiding overlapping.
    min_duration = 1  # Seconds
    hop_length_frame = sample_rate + 64
    n_fft = 255
    hop_length_fft = 63


    path_save_spectrogram   = 'Train/spectrogram/'
    path_save_time_serie    = 'Train/time_serie/'
    path_save_sound         = 'Train/sound/'
    path_train = "Train/finished_28spk_para_example/"



    audio_folders = sorted(os.listdir(path_train))  # get folder names
    print(audio_folders)
    clean_list = []
    noise_list = []
    for file_name in IncrementalBar('Processing').iter(audio_folders):
        print("\n{}\n".format(file_name))
        path_to_audio = path_train + file_name
        path_to_clean = path_to_audio +'/clean.wav'
        path_to_noise = path_to_audio + '/noise.wav'
        print("path to audio: {}".format(path_to_audio))
        print("path to clean: {}".format(path_to_clean))
        print("path to noise: {}".format(path_to_noise))
        clean_list.append(path_to_clean)
        noise_list.append(path_to_noise)

    # 1. Audio files to numpy
    # audio_dir = 'spectrogramVisualizing/All_together'
    audio_dir = ''
    # [START] AUDIO FILES TO NUMPY + SAVE LONG WAVES
    import glob
    # clean_list = [os.path.basename(x) for x in sorted(glob.glob("{}/clean*".format(audio_dir)))]
    # noise_list = [os.path.basename(x) for x in sorted(glob.glob("{}/noise*".format(audio_dir)))]
    # print(clean_list)
    # print(noise_list)

    # Squared spectrogram dimensions
    dim_square_spec = int(n_fft / 2) + 1

    # CLEAN VOICE
    clean_voice = audio_files_to_numpy(audio_dir, clean_list[6000:], sample_rate, frame_length, hop_length_frame, min_duration)
    save_audio(clean_voice.flatten(), sample_rate, "clean_long.wav")
    # Save to disk for Training / QC
    np.save(path_save_time_serie + 'voice_timeserie', clean_voice)
    # Create Amplitude and phase of the sounds
    m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            clean_voice, dim_square_spec, n_fft, hop_length_fft)
    np.save(path_save_spectrogram + 'voice_amp_db', m_amp_db_voice)
    np.save(path_save_spectrogram + 'voice_pha_db', m_pha_voice)



    # NOISE
    noise = audio_files_to_numpy(audio_dir, noise_list[6000:], sample_rate, frame_length, hop_length_frame, min_duration)
    save_audio(noise.flatten(), sample_rate, "noise_long.wav")
    # Save to disk for Training / QC
    np.save(path_save_time_serie + 'noise_timeserie', noise)
    # Create Amplitude and phase of the sounds
    m_amp_db_noise,  m_pha_noise = numpy_audio_to_matrix_spectrogram(
            noise, dim_square_spec, n_fft, hop_length_fft)
    np.save(path_save_spectrogram + 'noise_amp_db', m_amp_db_noise)
    np.save(path_save_spectrogram + 'noise_pha_db', m_pha_noise)


    # NOISY FILE
    noisy = clean_voice + noise
    print("shape of clean_voice: {}".format(clean_voice.shape))
    print("shape of noisy: {}".format(noisy.shape))
    print("shape of noise: {}".format(noise.shape))
    print("\n\n NaN in CLEAN: {}\n\n".format(np.isnan(clean_voice).any()))
    print("\n\n NaN in NOISE: {}\n\n".format(np.isnan(noise).any()))
    print("\n\n NaN in NOISY: {}\n\n".format(np.isnan(noisy).any()))

    if np.isnan(clean_voice).any():
        print(np.argwhere(np.isnan(clean_voice)))
    if np.isnan(noise).any():
        print(np.argwhere(np.isnan(noise)))
    if np.isnan(noisy).any():
        print(np.argwhere(np.isnan(noisy)))
    save_audio(noisy.flatten(), sample_rate, "noisy_long.wav")
    # Save to disk for Training / QC
    np.save(path_save_time_serie + 'noisy_voice_timeserie', noisy)
    # Create Amplitude and phase of the sounds
    m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            noisy, dim_square_spec, n_fft, hop_length_fft)
    np.save(path_save_spectrogram + 'noisy_voice_amp_db', m_amp_db_noisy_voice)
    np.save(path_save_spectrogram + 'noisy_voice_pha_db', m_pha_noisy_voice)


    print("shape of clean_voice: {}".format(clean_voice.shape))
    print("shape of noisy: {}".format(noisy.shape))
    print("shape of noise: {}".format(noise.shape))
    # [END] AUDIO FILES TO NUMPY + SAVE LONG WAVES


    '''
    # Display a spectrogram

    import matplotlib.pyplot as plt
    librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max), y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    '''









    # TODO: Dimensions of histogram
    # TODO: create spectrograms in numpy arrays
    # TODO: Save spectrograms on disk

if __name__== "__main__":
    main()



'''
NAN VALUES OF NOISY
[[10856  6347]
 [10856  6402]
 [10856  6403]
 [10856  6414]
 [10856  6415]
 [10856  6474]
 [10856  6475]
 [10856  6486]
 [10856  6487]
 [10856  6526]
 [10856  6527]
 [10856  6538]
 [10856  6539]
 [10856  6550]
 [10856  6551]
 [10856  6562]
 [10856  6563]
 [10856  6854]
 [10856  6855]
 [12471  1800]
 [12471  2124]
 [12471  2125]
 [12471  2132]
 [12677  3318]
 [12677  3319]
 [13421  2945]]

'''
