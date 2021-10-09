import os
from app import app
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2 as cv2
import random
import numpy as np
from skimage.feature import hog
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

orientations = 9
cellsPerBlock = 2
pixelsPerBlock = 16
convert_Color_Space = True

def Get_Features_From_Hog(image,orient,cellsPerBlock,pixelsPerCell, visualize= False, feature_vector_flag=True):
    if(visualize==True):
        hog_features, hog_image = hog(image, orientations=orient,
                          pixels_per_cell=(pixelsPerCell, pixelsPerCell),
                          cells_per_block=(cellsPerBlock, cellsPerBlock),
                          visualize=True, feature_vector=feature_vector_flag)
        return hog_features, hog_image
    else:
        hog_features = hog(image, orientations=orient,
                          pixels_per_cell=(pixelsPerCell, pixelsPerCell),
                          cells_per_block=(cellsPerBlock, cellsPerBlock),
                          visualize=False, feature_vector=feature_vector_flag)
        return hog_features


def Extract_Features(images, orientation, cellsPerBlock, pixelsPerCell, convert_Color_Space=False):
    feature_List = []
    image_List = []
    for image in images:
        if (convert_Color_Space == True):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        local_features_one = Get_Features_From_Hog(image[:, :, 0], orientation, cellsPerBlock, pixelsPerCell, False,
                                                   True)
        local_features_two = Get_Features_From_Hog(image[:, :, 1], orientation, cellsPerBlock, pixelsPerCell, False,
                                                   True)
        local_features_three = Get_Features_From_Hog(image[:, :, 2], orientation, cellsPerBlock, pixelsPerCell, False,
                                                     True)
        x = np.hstack((local_features_one, local_features_two, local_features_three))
        feature_List.append(x)
    return feature_List


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def DrawCars(image, windows, converColorspace=False):
    refined_Windows = []
    for window in windows:

        start = window[0]
        end = window[1]
        clippedImage = image[start[1]:end[1], start[0]:end[0]]

        if (clippedImage.shape[1] == clippedImage.shape[0] and clippedImage.shape[1] != 0):

            clippedImage = cv2.resize(clippedImage, (64, 64))

            f1 = Extract_Features([clippedImage], 9, 2, 16, converColorspace)

            # print(len(predictedOutput))
            pickle_in = open('model/training.pickle', "rb")
            model = pickle.load(pickle_in)
            predictedOutput=model.predict([f1[0]])
            # print("sss", model)
            # predictedOutput=classifier_one.predict([f1[0]])

            if (predictedOutput == 1):
                # print("Nice car!")#Add. aboolena statement.
                refined_Windows.append(window)

    return refined_Windows

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    image_copy = np.copy(img)
    for bbox in bboxes:
        r=random.randint(0,255)
        g=random.randint(0,255)
        b=random.randint(0,255)
        color=(r, g, b)
        cv2.rectangle(image_copy, bbox[0], bbox[1], color, thick)
    return image_copy

def slide_window(img, xstart_stop=[None, None], ystart_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.9, 0.9)):
    if xstart_stop[0] == None:
        xstart_stop[0] = 0
    if xstart_stop[1] == None:
        xstart_stop[1] = img.shape[1]
    if ystart_stop[0] == None:
        ystart_stop[0] = 0
    if ystart_stop[1] == None:
        ystart_stop[1] = img.shape[0]

    window_list = []
    image_width_x = xstart_stop[1] - xstart_stop[0]
    image_width_y = ystart_stop[1] - ystart_stop[0]

    windows_x = np.int(1 + (image_width_x - xy_window[0]) / (xy_window[0] * xy_overlap[0]))
    windows_y = np.int(1 + (image_width_y - xy_window[1]) / (xy_window[1] * xy_overlap[1]))

    modified_window_size = xy_window
    for i in range(0, windows_y):
        y_start = ystart_stop[0] + np.int(i * modified_window_size[1] * xy_overlap[1])
        for j in range(0, windows_x):
            x_start = xstart_stop[0] + np.int(j * modified_window_size[0] * xy_overlap[0])

            x1 = np.int(x_start + modified_window_size[0])
            y1 = np.int(y_start + modified_window_size[1])
            window_list.append(((x_start, y_start), (x1, y1)))
    return window_list

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    image = mpimg.imread('static/uploads/'+filename)

    window_one = slide_window(image, xstart_stop=[0, 1280], ystart_stop=[400, 464],
                              xy_window=(64, 64), xy_overlap=(0.8, 0.8))

    window_two = slide_window(image, xstart_stop=[0, 1280], ystart_stop=[400, 612],
                              xy_window=(96, 96), xy_overlap=(0.8, 0.8))

    window_three = slide_window(image, xstart_stop=[0, 1280], ystart_stop=[400, 660],
                                xy_window=(128, 128), xy_overlap=(0.8, 0.8))

    window_four = slide_window(image, xstart_stop=[0, 1280], ystart_stop=[400, 480],
                               xy_window=(80, 80), xy_overlap=(0.8, 0.8))

    windows = window_one + window_two + window_three + window_four
    # windows = window_two # +  window_three + window_four

    print("Total Number of windows are ", len(windows))
    refined_Windows = DrawCars(image, windows, True)

    # window_img = draw_boxes(image, windows)

    window_img = draw_boxes(image, refined_Windows)
    print(type(window_img))
    cv2.imwrite('static/predicted/test.png', window_img)

    return redirect(url_for('static', filename='predicted/' + "test.png"), code=301)


if __name__ == "__main__":
    app.run()