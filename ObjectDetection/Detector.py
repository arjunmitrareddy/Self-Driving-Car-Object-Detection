from ObjectDetection.SlidingWindowHelper import slide_window, search_windows, add_heat, apply_threshold, draw_labeled_bboxes, draw_boxes
from ObjectDetection.TrainPhase import load_classifier
from ObjectDetection.ImageReader import read_images
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

LOAD_SAVED_MODEL = True

clf, X_scaler = load_classifier(LOAD_SAVED_MODEL)
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 1  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (48, 48)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

images = read_images('../test_images')

def pipeline(image):
    draw_image = np.copy(image)
    image = image.astype(np.float32) / 255
    windows = []
    sizes = [(64, 64), (96, 96), (128, 128), (256, 256)]
    for size in sizes:
        windows += slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[400, 656],
                                xy_window=size, xy_overlap=(0.75, 0.75))

    hot_windows = search_windows(image, windows, clf, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    plt
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    draw_img = draw_labeled_bboxes(draw_image, labels)
    return draw_img

output = '../demo.mp4'
clip = VideoFileClip('../project_video.mp4')
white_clip = clip.fl_image(pipeline)
white_clip.write_videofile(output, audio=False)