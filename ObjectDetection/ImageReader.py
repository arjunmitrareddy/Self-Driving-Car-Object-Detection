import glob
import matplotlib.image as mpimg

def read_images(path):
    images = []

    for filename in glob.iglob(path + "/**/*.*", recursive=True):
        images.append(mpimg.imread(filename))

    return images