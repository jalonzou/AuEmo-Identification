import librosa
import librosa.display
import librosa.core
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import matplotlib

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

def display_features(feature, sr=44010):
    # Make a new figure
    plt.figure(figsize=(25,8))

    # Display the spectrogram on a mel scale

    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(feature, sr=sr, x_axis='time', y_axis='mel',cmap='gray',fmax=8000)

    # Put a descriptive title on the plot
    #plt.title('mel power spectrogram')

    # draw a color bar
    #plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('sample.jpg')
    plt.show()


def extract_features(parent_dir, sub_dirs, file_ext="*.mp3", bands = 128, frames = 128):
    for l, sub_dir in enumerate(sub_dirs):
        sub_mv_dir = sub_dir[:-1] + '_test'
        mv_dir = os.path.join(parent_dir, sub_mv_dir)
        num_count = 0
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            if num_count == 140: break

            gn = os.path.join("img_db_test",sub_dir)
            
            seq = [gn,sub_dir[0:-1], '_%03d'%num_count,".png"]
            gn = ''.join(seq)
            print(gn)
            sound_clip, sr = librosa.load(fn)
            #D = np.abs(librosa.stft(sound_clip))**2
            melspec = librosa.feature.melspectrogram(y=sound_clip, n_mels=128)
            #print (melspec.shape)
            logspec = librosa.amplitude_to_db(melspec)
            test = np.flip(logspec,axis=0)

            if np.size(test, 1) > 2000:
                test = test[:, 700:1700]
            elif np.size(test, 1) > 1000:
                test = test[:, 200:1200]
            else: continue

            print(test.shape)
            matplotlib.image.imsave(gn, test)

            os.rename(fn, os.path.join(mv_dir, 'valid_%03d.mp3'%num_count))

            num_count = num_count + 1

def main():
    parent_dir = "song_emotion_db/"
    # sub_dir = ["angry/","happy/","horror/","sad/","peaceful/"]
    sub_dir = ["sad/","angry/"]
    features = extract_features(parent_dir, sub_dir)


if __name__ == '__main__':
    main()
