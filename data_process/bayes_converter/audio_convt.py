import librosa
import librosa.display
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(parent_dir, sub_dirs, file_ext="*.mp3", bands = 128, frames = 128):
    window_size = 512 * 127
    log_specgrams = []
    labels = []
    cur_label = -1
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, sr = librosa.load(fn)
            print fn
            print sr
            cur_label += 1
            for (start, end) in windows(sound_clip, window_size):
                if len(sound_clip[start:end]):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    print melspec.shape
                    logspec = librosa.amplitude_to_db(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(cur_label)
    while not log_specgrams[-1].shape[1] == 16384:
        log_specgrams.pop()

    log_specgrams = np.array(log_specgrams)
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames)
    print log_specgrams.shape
    features = log_specgrams
    return np.array(features)

def display_features(feature, sr=22050):
    # Make a new figure
    plt.figure(figsize=(12,8))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(feature, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()

    plt.show()

def main(argv=None):
    parent_dir = "/Users/zjn/Documents/BU_courses/CS_ML/FINAL/dataset/"
    sub_dir = ["emo1/"]
    features = extract_features(parent_dir, sub_dir, bands = 128, frames = 128)
    display_features(features[1])

if __name__ == '__main__':
    main()
