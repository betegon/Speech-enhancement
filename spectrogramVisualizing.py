import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


# TODO: Document That I have used zero padding for STFT
# TODO: Change values in args.py to get 16064Hz as frame rate  and all the other parameters.
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
    for file in list_audio_files:
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

    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)

    return stftaudio_magnitude_db, stftaudio_phase


def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    """This function takes as input a numpy audio of size (nb_frame,frame_length), and return
    a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
    (nb_frame,dim_square_spec,dim_square_spec)"""
    print("Numpy audio size: {}".format(numpy_audio.shape))
    nb_audio = numpy_audio.shape[0]

    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i])

    return m_mag_db, m_phase




def save_audio(y, sample_rate, output_name='audio_ouput.wav'):
    '''
    Save audio file given y (amplitude values) and sample_rate.
    By default, output name is 'audio_output.wav'

    Args:
        y (ndarray): shape should be (n,) or (2,n).
    '''
    librosa.output.write_wav(output_name, y, sample_rate)


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
    clean_filename = 'clean.wav'
    mixed_filename = 'mixed.wav'
    noise_filename = 'noise.wav'
    sample_rate = 16000  # Hz
    frame_length = sample_rate + 128  # a bit more than sample_rate for avoiding overlapping.
    min_duration = 1  # Seconds
    hop_length_frame = sample_rate + 128
    n_fft = 512
    hop_length_fft = 63
    # Load .wav files to plot
    y_clean, sr_clean = librosa.load(os.path.join(audio_dir, clean_filename), sr=sample_rate)
    y_mixed, sr_mixed = librosa.load(os.path.join(audio_dir, mixed_filename), sr=sample_rate)
    y_noise, sr_noise = librosa.load(os.path.join(audio_dir, noise_filename), sr=sample_rate)
    # Total duration = y/sr
    print("Duration clean: {} seconds.".format(librosa.get_duration(y=y_clean, sr=sr_clean)))
    print("Duration mixed: {} seconds.".format(librosa.get_duration(y=y_mixed, sr=sr_mixed)))
    print("Duration noise: {} seconds.".format(librosa.get_duration(y=y_noise, sr=sr_noise)))

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
    audio_dir = 'spectrogramVisualizing/All_together'
    clean1 = 'clean.wav'
    clean2 = 'clean2.wav'
    y_clean1, sr_clean1 = librosa.load(os.path.join(audio_dir, clean1), sr=sample_rate)
    y_clean2, sr_clean2 = librosa.load(os.path.join(audio_dir, clean2), sr=sample_rate)
    print("Duration clean1: {} seconds.".format(librosa.get_duration(y=y_clean1, sr=sr_clean1)))
    print("Duration clean2: {} seconds.".format(librosa.get_duration(y=y_clean2, sr=sr_clean2)))

    # [START] AUDIO FILES TO NUMPY + SAVE LONG WAVES
    clean_voice = audio_files_to_numpy(audio_dir, ['clean.wav', 'clean2.wav'], sample_rate, frame_length, hop_length_frame, min_duration)
    noise = audio_files_to_numpy(audio_dir, ['noise.wav', 'noise2.wav'], sample_rate, frame_length, hop_length_frame, min_duration)
    noisy = clean_voice + noise
    print("shape of clean_voice: {}".format(clean_voice.shape))
    print("shape of noisy: {}".format(noisy.shape))
    print("shape of noise: {}".format(noise.shape))
    save_audio(clean_voice.flatten(),sample_rate,"clean_long.wav")
    save_audio(noisy.flatten(),sample_rate,"noisy_long.wav")
    save_audio(noise.flatten(),sample_rate,"noise_long.wav")
    # [END] AUDIO FILES TO NUMPY + SAVE LONG WAVES






    # Squared spectrogram dimensions
    dim_square_spec = int(n_fft / 2) + 1

    # Create Amplitude and phase of the sounds
    m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            clean_voice, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noise,  m_pha_noise = numpy_audio_to_matrix_spectrogram(
            noise, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            noisy, dim_square_spec, n_fft, hop_length_fft)

    '''
    # Save to disk for Training / QC
    np.save(path_save_time_serie + 'voice_timeserie', prod_voice)
    np.save(path_save_time_serie + 'noise_timeserie', prod_noise)
    np.save(path_save_time_serie + 'noisy_voice_timeserie', prod_noisy_voice)


    np.save(path_save_spectrogram + 'voice_amp_db', m_amp_db_voice)
    np.save(path_save_spectrogram + 'noise_amp_db', m_amp_db_noise)
    np.save(path_save_spectrogram + 'noisy_voice_amp_db', m_amp_db_noisy_voice)

    np.save(path_save_spectrogram + 'voice_pha_db', m_pha_voice)
    np.save(path_save_spectrogram + 'noise_pha_db', m_pha_noise)
    np.save(path_save_spectrogram + 'noisy_voice_pha_db', m_pha_noisy_voice)
    '''





    # TODO: Dimensions of histogram
    # TODO: create spectrograms in numpy arrays
    # TODO: Save spectrograms on disk

if __name__== "__main__":
    main()
