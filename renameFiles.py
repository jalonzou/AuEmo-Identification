
import os
from PIL import Image

rootdir = "piano_emotion_db/"


def main():
    for subdir, dirs, files in os.walk(rootdir):
        i = 0;
        for file in files:
            # print(os.path.split(subdir)[1]) ## angry
            if(file.endswith("mp3")):
                # print(os.path.join(subdir, file))
                newName = os.path.split(subdir)[1] + '_' + str(i).zfill(3) + '.mp3' ## angry_001.mp3
                i += 1               
                os.rename(os.path.join(subdir, file),
                  os.path.join(subdir, newName))


if  __name__ =='__main__':
    print("rename...")
    main()





