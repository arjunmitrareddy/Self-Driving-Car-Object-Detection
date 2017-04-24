from ObjectDetection.ImageReader import read_images
from ObjectDetection.FeatureExtraction.FeatureExtractor import extract_features
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import numpy as np
import time

LOAD_SAVED_IMAGES = False
SAVE_MODEL = True
LOAD_SCALED_FEATURES = False
LOAD_SAVED_FEATURES = False
PATHS = {
    "images" : {
        "cars_read": '../vehicles',
        "notcars_read": '../non-vehicles',
        "cars_save": '../cars_save.npy',
        "not_cars_save": '../not_cars_save.npy'
    },
    "features": {
        "normal": {
            "car": '../car_features.npy',
            "not_car": '../not_car_features.npy'
        },
        "scaled": {
            "X": '../scaled_X.npy',
            "y": '../scaled_Y.npy'
        }
    },
    "model": '../svcModel.joblib.pkl',
    "scaler": '../svcScaled.joblib.pkl'
}

def load_classifier(load_model):
    if load_model:
        return joblib.load(PATHS["model"]), joblib.load(PATHS["scaler"])
    else:
        if not LOAD_SAVED_IMAGES:
            cars = read_images(PATHS["images"]["cars_read"])
            notcars = read_images(PATHS["images"]["notcars_read"])
            np.save(PATHS["images"]["cars_save"], cars)
            np.save(PATHS["images"]["not_cars_save"], notcars)
        else:
            cars = np.load(PATHS["images"]["cars_save"])
            notcars = np.load(PATHS["images"]["not_cars_save"])

        print(len(cars), len(notcars))

        color_space_cars = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        color_space_notcars = 'YCrCb'
        orient = 12  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 1 # HOG cells per block
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        spatial_size = (48, 48) # Spatial binning dimensions
        hist_bins = 32    # Number of histogram bins
        spatial_feat = True # Spatial features on or off
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off

        if not LOAD_SAVED_FEATURES:
            car_features = extract_features(cars, color_space=color_space_cars,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
            notcar_features = extract_features(notcars, color_space=color_space_notcars,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
            np.save(PATHS["features"]["normal"]["car"], car_features)
            np.save(PATHS["features"]["normal"]["not_car"], notcar_features)
        else:
            car_features = np.load(PATHS["features"]["normal"]["car"])
            notcar_features = np.load(PATHS["features"]["normal"]["not_car"])

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        if SAVE_MODEL:
            joblib.dump(X_scaler, PATHS["scaler"], compress=9)
        # Apply the scaler to X
        if not LOAD_SCALED_FEATURES:
            scaled_X = X_scaler.transform(X)
            y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
            np.save(PATHS["features"]["scaled"]["X"], scaled_X)
            np.save(PATHS["features"]["scaled"]["y"], y)
        else:
            scaled_X = np.load(PATHS["features"]["scaled"]["X"])
            y = np.load(PATHS["features"]["scaled"]["y"])

        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)




        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC

        svc = LinearSVC(loss='hinge')

        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        if SAVE_MODEL:
            joblib.dump(svc, PATHS["model"], compress=9)
        return svc, X_scaler