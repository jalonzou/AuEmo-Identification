from PIL import Image
import glob
import os

def convert(parent_dir,sub_dirs,file_ext="*.jpg"): 
	for l,sub_dir in enumerate(sub_dirs):
		for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
			im = Image.open(fn).convert('L')
			print(im)
			im.save(fn)


def main():
    parent_dir = "img_db/"
    sub_dir = ["angry/","happy/","horror/","sad/","peaceful/"]
    convert(parent_dir, sub_dir)


if __name__ == '__main__':
    main()
