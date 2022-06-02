# ================================================================
#
#   File name   : utils.py
#   Author      : Jeet
#   Description : additional yolov3 and yolov4 functions
#
# ================================================================
import os.path
from multiprocessing import Process, Queue, Pipe
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolov3.configs import *
from yolov3.yolov4 import *
from tensorflow.python.saved_model import tag_constants
import pytesseract
import easyocr
import base64
from pprint import pprint

def load_yolo_weights(model, weights_file):
    tf.keras.backend.clear_session()  # used to reset layer names
    # load Darknet original weights to TensorFlow model
    if YOLO_TYPE == "yolov3":
        range1 = 75 if not TRAIN_YOLO_TINY else 13
        range2 = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]
    if YOLO_TYPE == "yolov4":
        range1 = 110 if not TRAIN_YOLO_TINY else 21
        range2 = [93, 101, 109] if not TRAIN_YOLO_TINY else [17, 20]

    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' % i
            else:
                conv_layer_name = 'conv2d'

            if j > 0:
                bn_layer_name = 'batch_normalization_%d' % j
            else:
                bn_layer_name = 'batch_normalization'

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'


def Load_Yolo_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f'GPUs {gpus}')
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    if YOLO_FRAMEWORK == "tf":  # TensorFlow detection
        if YOLO_TYPE == "yolov4":
            Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
        if YOLO_TYPE == "yolov3":
            Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

        if YOLO_CUSTOM_WEIGHTS == True:
            print("Loading Darknet_weights from:", Darknet_weights)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
            load_yolo_weights(yolo, Darknet_weights)  # use Darknet weights
        else:
            checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
            if TRAIN_YOLO_TINY:
                checkpoint += "_Tiny"
            print("Loading custom weights from:", checkpoint)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            yolo.load_weights(checkpoint)  # use custom weights

    elif YOLO_FRAMEWORK == "trt":  # TensorRT detection
        saved_model_loaded = tf.saved_model.load(YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        yolo = saved_model_loaded.signatures['serving_default']

    return yolo


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes



'''def Image_to_Text(image):

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ''# now to read the  file
    image = cv2.imread(image)  # 002, 003, image4, image3,OIP
    print("Original image is reading..")
    # we will resize and standardise our image to 500
    image = imutils.resize(image, width=500)

    # we will display original image when it will start finding

    print("Original image is loading..")
    cv2.imshow("Original Image", image)  # original image is the name of window
    cv2.waitKey(0)  # till you press anyting it will not execute further

    # Now we will convert image to grey scale
    # Because it will reduce the dimensions and also reduce the complexity of image
    # and moreover there are few algo. like "canny" which works only on gray scale
    print("Image is converting to grayscale..")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Scale Image", gray)
    cv2.waitKey(0)

    # Now we will reduce noise from our image and make it smooth
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("Smoother Image", gray)
    cv2.waitKey(0)

    # So we will find the edges of images

    edged = cv2.Canny(gray, 10, 180)
    cv2.imshow("Canny edge", edged)
    cv2.waitKey(0)

    # Now we will find the cotours based on the images

    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts is contours which is like the curve joining all the contours points
    # new is heirarchy - relationship
    # RETR_LISt retrives all the contours but doesn;t create any parents-child relationship
    # Chain approx points removes all the redundant points and compress the contours by saving memory

    # we will create a copy of our original image to draw all the contours

    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)  # this values are fixed
    cv2.imshow("Canny after contouring", image1)
    cv2.waitKey(0)

    # Now we don't want all the contours, we are interested only in number plate
    # but can't directly locate that so we will sort them on the basis of their areas
    # we will select those area which are max. so we will select top 30 areas
    # but it will give sorted list as in order of min to max
    # so for that we will reverse the order of

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0:10]

    NumberPlateCount = None
    for contour in cnts:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            NumberPlateCount = approx
            break

    # because currently we don't have any contours
    # to draw top30 contours we will make copy of original image and use because we don't want edit the original image
    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("Top30 Contours", image2)
    cv2.waitKey(0)

    # now we will run a for loop on our contours to find the best possible contours of our expectes no. plate
    count = 0
    name = 1  # name of our image(cropped image)

    for i in cnts:
        perimeter = cv2.arcLength(i, True)
        # perimeter is called arclength and we directly use it's function
        approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
        # approxpolydp approximates the curve of polygon
        if (len(approx) == 4):  # 4 because plate has 4 corners
            NumberPlateCount = approx
            # print(NumberPlateCount)
            # now we crop that rectangle part
            x, y, w, h = cv2.boundingRect(NumberPlateCount)
            crp_image = image[y:y + h, x:x + w]

            ###################
            #                 #
            # (y+h)   #                 #
            ###################
            # (x,y)         #(x+w)

            cv2.imwrite(str(name) + '.png', crp_image)
            name += 1
            break

    # Now we will draw contour in our main image that we have identified as a number plate

    cv2.drawContours(image, [NumberPlateCount], 0, (0, 255, 0), 3)
    cv2.imshow("Final Image", image)
    print("Final image is printed")
    cv2.waitKey(0)

    # we have got our no. plate in main image

    # we will crop only the no. plate part

    #crop_img_loc = './IMAGES/1.png'
    #cv2.imshow("Cropped Image", cv2.imread(crop_img_loc))
    #print("License plate is cropped")
    reader = easyocr.Reader(['en', 'en'])  # need to run only once to load model into memory [english(en) to english(en) convertion]
    result = reader.readtext(ocr)
    print(result)
    '''


def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        # print(x1,y1,x2,y2)
        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " " + str(score)

            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
                # print("asjdgauy")
                # print(label)
                #
                # # get text size
                # (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale,
                #                                                       thickness=bbox_thick)
                # # put filled text rectangle
                # cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                #               thickness=cv2.FILLED)
                #
                # # put text above rectangle
                # cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #             fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                         fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
#################################################################################################################################################
            # ocr = image[y1:y2, x1:x2]
            # cv2.imshow('ocr',ocr)
            # cv2.imshow('license_plate',ocr)
            # reader = easyocr.Reader(['en', 'en'])  # need to run only once to load model into memory [english(en) to english(en) convertion]
            # result = reader.readtext(ocr)
            # #result = pytesseract.image_to_string(ocr)
            # #print(result)
            # #put text above rectangle
            # cv2.putText(image, str((result[0][1], round(result[0][2], 2))), (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

##################################################################################################################################################
    return image
######################################################


def draw_bbox2(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    y = 1

    x1, y1 = None, None

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # print(x1,y1,x2,y2)

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        # lic_plates = []
        # lic_plates.append(image[y1:y2, x1:x2])
        #
        # file = r'D:\backup\license_plate_detection\TensorFlow-2.x-YOLOv3\IMAGES\Cropped'
        #
        #
        # for x in lic_plates:
        #     # print(x)
        #     cv2.imshow(f'CROPPED_{y}', x)
        #     cv2.imwrite(fr"D:\backup\license_plate_detection\TensorFlow-2.x-YOLOv3\IMAGES\Cropped\cropped_{y}.jpg", x)  # for image detection
        #     y += 1
        #
        # cv2.waitKey(1)
        croppedlicenseplate = image[y1:y2, x1:x2]
        # cv2.imwrite("./IMAGES/croppedlicenseplate.jpg",croppedlicenseplate)
        # cv2.imwrite("./IMAGES/croppedlicenseplate.jpg", image)  # for image detection

        cv2.imwrite("./IMAGES/croppedlicenseplate.jpg",croppedlicenseplate)

        # cv2.imshow("Croppedlicenseplate",croppedlicenseplate)
        # cv2.waitKey(0)

        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # image = frame[y1:y2, x1:x2]

        import boto3
        client = boto3.client('textract')
        imgfilename = r"./IMAGES/croppedlicenseplate.jpg"
        # print("hello")
        # def get_file_from_filepath(filename):
        with open(imgfilename, 'rb') as imgfile:
            imageBytes = bytearray(imgfile.read())
            # print(type(imageBytes))

        # imgbyte = get_file_from_filepath(imgfilename)

        result = client.detect_document_text(Document={'Bytes': imageBytes})
        shri = result['Blocks']

        text_arr = []

        for item in shri:
            if item["BlockType"] == "LINE":
                text_arr.append(item['Text'])
            # else:
            #     print("Text is not clear")

        # print(text_arr)
        # cv2.putText(image, str(text_arr), (50, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=(0, 255, 0),
        #             thickness=1)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " " + str(score)

            try:
                # label = "{}".format(NUM_CLASS[class_ind]) + score_str
                for i in text_arr:
                    label = i
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                          fontScale,
                                                                          thickness=bbox_thick)
                    # print(text_height, text_width, baseline, 'HIIII')
                    # put filled text rectangle
                    cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                                  thickness=cv2.FILLED)

                    # put text above rectangle
                    cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors,
                                bbox_thick, lineType=cv2.LINE_AA)
                # print("asjdgauy")
                # print(label)
                #
                # # get text size
                # (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale,
                #                                                       thickness=bbox_thick)
                # # put filled text rectangle
                # cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                #               thickness=cv2.FILLED)
                #
                # # put text above rectangle
                # cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #             fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

                # get text size


                # cv2.putText(image, text_arr[0], (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)


            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # # get text size
            #
            # (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
            # print(text_height, text_width, baseline, 'HIIII')
            # # put filled text rectangle
            # cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
            #
            # # put text above rectangle
            # cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
            # # cv2.putText(image, text_arr[0], (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)





#################################################################################################################################################
            # ocr = image[y1:y2, x1:x2]
            # cv2.imshow('ocr',ocr)
            # cv2.imshow('license_plate',ocr)
            # reader = easyocr.Reader(['en', 'en'])  # need to run only once to load model into memory [english(en) to english(en) convertion]
            # result = reader.readtext(ocr)
            # #result = pytesseract.image_to_string(ocr)
            # #print(result)
            # #put text above rectangle
            # cv2.putText(image, str((result[0][1], round(result[0][2], 2))), (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

##################################################################################################################################################

    return image, x1, y1
    # return croppedlicenseplate

#####################################


def draw_bbox_3(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        # print(x1,y1,x2,y2)
        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " " + str(score)

            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
                # print("asjdgauy")
                # print(label)
                #
                # # get text size
                # (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale,
                #                                                       thickness=bbox_thick)
                # # put filled text rectangle
                # cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                #               thickness=cv2.FILLED)
                #
                # # put text above rectangle
                # cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #             fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
#################################################################################################################################################
            # ocr = image[y1:y2, x1:x2]
            # cv2.imshow('ocr',ocr)
            # cv2.imshow('license_plate',ocr)
            # reader = easyocr.Reader(['en', 'en'])  # need to run only once to load model into memory [english(en) to english(en) convertion]
            # result = reader.readtext(ocr)
            # #result = pytesseract.image_to_string(ocr)
            # #print(result)
            # #put text above rectangle
            # cv2.putText(image, str((result[0][1], round(result[0][2], 2))), (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

##################################################################################################################################################
    return image



def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def detect_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 score_threshold=0.45, iou_threshold=0.45, rectangle_colors=''):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    # print(bboxes)

    draw_bbox2(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)



    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    ######################################

    import boto3
    # import numpy as np
    # import requests
    # from pprint import pprint

    # client = boto3.client('rekognition')
    client = boto3.client('textract')

    imgfilename = r"./IMAGES/croppedlicenseplate.jpg"

    # def get_file_from_filepath(filename):
    with open(imgfilename, 'rb') as imgfile:
        imageBytes = bytearray(imgfile.read())
        # print(type(imageBytes))

    # imgbyte = get_file_from_filepath(imgfilename)

    result = client.detect_document_text(Document={'Bytes': imageBytes})
    shri = result['Blocks']

    text_arr = []

    for item in shri:
        if item["BlockType"] == "LINE":
            text_arr.append(item['Text'])
        # else:
        #     print("Text is not clear")

    print(text_arr)

    ######################################




    # cv2.imshow('img',)
    # draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
    # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

    # if output_path != '': cv2.imwrite(output_path, image)

    ###############################################################################
    # image_path = image
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

    # image = cv2.imread(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # reader = pytesseract.image_to_string(image, lang='eng')
    # print(reader)

    # reader = easyocr.Reader(['en', 'en'])
    # result = reader.readtext(img)
    # result = reader.readtext(original_image)
    # result = reader.readtext(bboxes)
    # pprint(result)

    # Save image
    # if output_path != '': cv2.imwrite(output_path, image)
    if output_path != '': cv2.imwrite(output_path, original_image)

    # cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #             fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
    #
    # NUM_CLASS = read_class_names(CLASSES)
    # num_classes = len(NUM_CLASS)
    # image_h, image_w, _ = image.shape
    # hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # # print("hsv_tuples", hsv_tuples)
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    #
    # image_h, image_w,_ = image.shape
    # for i, bbox in enumerate(bboxes):
    #     coor = np.array(bbox[:4], dtype=np.int32)
    #     score = bbox[4]
    #
    #     class_ind = int(bbox[5])
    #     bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
    #     bbox_thick = int(0.6 * (image_h + image_w) / 1000)
    #
    #     fontScale = 0.75 * bbox_thick
    #     (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
    #     cv2.putText(image, result, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick, lineType=cv2.LINE_AA)
    ###############################################################################

    if show:
        # Show the image
        cv2.imshow("predicted image", original_image)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()

    return original_image


def Predict_bbox_mp(Frames_data, Predicted_data, Processing_times):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")
    Yolo = Load_Yolo_model()
    times = []
    while True:
        if Frames_data.qsize() > 0:
            image_data = Frames_data.get()
            t1 = time.time()
            Processing_times.put(time.time())

            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            Predicted_data.put(pred_bbox)
            print(Predicted_data)


def postprocess_mp(Predicted_data, original_frames, Processed_frames, Processing_times, input_size, CLASSES,
                   score_threshold, iou_threshold, rectangle_colors, realtime):
    times = []
    while True:
        if Predicted_data.qsize() > 0:
            pred_bbox = Predicted_data.get()
            if realtime:
                while original_frames.qsize() > 1:
                    original_image = original_frames.get()
            else:
                original_image = original_frames.get()

            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')
            image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            times.append(time.time() - Processing_times.get())
            times = times[-20:]

            ms = sum(times) / len(times) * 1000
            fps = 1000 / ms
            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255), 2)
            # print("Time: {:.2f}ms, Final FPS: {:.1f}".format(ms, fps))

            Processed_frames.put(image)


def Show_Image_mp(Processed_frames, show, Final_frames):
    while True:
        if Processed_frames.qsize() > 0:
            image = Processed_frames.get()
            Final_frames.put(image)
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break


# detect from webcam
def detect_video_realtime_mp(video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                             score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', realtime=False):
    if realtime:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4
    no_of_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    original_frames = Queue()
    Frames_data = Queue()
    Predicted_data = Queue()
    Processed_frames = Queue()
    Processing_times = Queue()
    Final_frames = Queue()

    p1 = Process(target=Predict_bbox_mp, args=(Frames_data, Predicted_data, Processing_times))
    p2 = Process(target=postprocess_mp, args=(
    Predicted_data, original_frames, Processed_frames, Processing_times, input_size, CLASSES, score_threshold,
    iou_threshold, rectangle_colors, realtime))
    p3 = Process(target=Show_Image_mp, args=(Processed_frames, show, Final_frames))
    p1.start()
    p2.start()
    p3.start()

    while True:
        ret, img = vid.read()
        if not ret:
            break

        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_frames.put(original_image)

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        Frames_data.put(image_data)

    while True:
        if original_frames.qsize() == 0 and Frames_data.qsize() == 0 and Predicted_data.qsize() == 0 and Processed_frames.qsize() == 0 and Processing_times.qsize() == 0 and Final_frames.qsize() == 0:
            p1.terminate()
            p2.terminate()
            p3.terminate()
            break
        elif Final_frames.qsize() > 0:
            image = Final_frames.get()
            if output_path != '': out.write(image)

    cv2.destroyAllWindows()


def detect_video(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,  score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times, times_2 = [], []
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))              # 2562
    width = 1280
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))            # 1440
    height = 720
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc(*'XVID')
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, 25, (width, height))  # output_path must be .mp4

    while True:
        ret, frame = vid.read()

        img = cv2.resize(frame, (1280, 720))
        img_h, img_w, img_c = img.shape                     # height, width, channel

        # print(img_h,img_w)

            # halfHeight = int(img_h / 2)
            # onethirdHeight = int(2 * img_h/ 3)
            # new_img = img[onethirdHeight:img_h, 0:img_w]

            # cv2.imshow('new', new_img)


            # print(img_w, img_h)
            # print(img.shape)

        if ret:
            try:
                original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                original_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            except:
                break

            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            t1 = time.time()
            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)

            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)
            # print("before", pred_bbox)
            t2 = time.time()

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            # print("after",pred_bbox)
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)

            #bboxes = postprocess_boxes(pred_bbox, new_img, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')
            image, x_pos, y_pos = draw_bbox2(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            # image = draw_bbox(new_img, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            # image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            # print(type(image))

            t3 = time.time()
            times.append(t2 - t1)
            times_2.append(t3 - t1)

            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times) / len(times) * 1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2) / len(times_2) * 1000)

            ##############################################################################################
            # ocr = image[width : width + width, height: height + height]
            # cv2.imshow('ocr', ocr)
            # # cv2.imshow('license_plate',ocr)
            # reader = easyocr.Reader(
            #     ['en', 'en'])  # need to run only once to load model into memory [english(en) to english(en) convertion]
            # result = reader.readtext(ocr)
            #
            # # put text above rectangle
            # cv2.putText(image, str((result[0][1], round(result[0][2], 2))), (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #             fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
            ##############################################################################################

            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            # cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)


            # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
            #
            # fps = int(vid.get(cv2.CAP_PROP_FPS))
            # # codec = cv2.VideoWriter_fourcc(*'XVID')
            # codec = cv2.VideoWriter_fourcc(*'XVID')
            # # print(img_w,img_h)
            # out = cv2.VideoWriter(output_path, codec, fps, (img_w, img_h))  # output_path must be .mp4

            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
            #
            # import boto3
            # client = boto3.client('textract')
            # imgfilename = r"./IMAGES/croppedlicenseplate.jpg"
            # # print("hello")
            # # def get_file_from_filepath(filename):
            # with open(imgfilename, 'rb') as imgfile:
            #     imageBytes = bytearray(imgfile.read())
            #     # print(type(imageBytes))
            #
            # # imgbyte = get_file_from_filepath(imgfilename)
            #
            # result = client.detect_document_text(Document={'Bytes': imageBytes})
            # shri = result['Blocks']
            #
            # text_arr = []
            #
            # for item in shri:
            #     if item["BlockType"] == "LINE":
            #         text_arr.append(item['Text'])
            #     # else:
            #     #     print("Text is not clear")
            #
            # print(text_arr)
            # cv2.putText(image, str(text_arr), (50, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=(0, 255, 0), thickness=1)

            # import boto3
            # client = boto3.client('textract')
            # imgfilename = r"./IMAGES/croppedlicenseplate.jpg"
            # for images in os.listdir('D://backup//license_plate_detection//TensorFlow-2.x-YOLOv3//IMAGES//Cropped'):
            #     imgfilename = images
            #     # print("hello")
            #     # def get_file_from_filepath(filename):
            #
            #     # with open(imgfilename, 'rb') as imgfile:
            #     #     imageBytes = bytearray(imgfile.read())
            #
            #         # print(type(imageBytes))

            # imgbyte = get_file_from_filepath(imgfilename)

            # result = client.detect_document_text(Document={'Bytes': imageBytes})
            # shri = result['Blocks']

            text_arr = []

            # for item in shri:
            #     if item["BlockType"] == "LINE":
            #         text_arr.append(item['Text'])
                # else:
                #     print("Text is not clear")

            # print(text_arr)
            # print(type(x_pos), type(y_pos))
            # cv2.putText(image, str(text_arr), (50, 200),cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale= 1.0, color=(0, 255, 0), thickness=1 )
            # cv2.putText(image, str('HI'), (int(x_pos) + 10, int(y_pos) + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale= 1.0, color=(0, 255, 0), thickness=1 )
            # cv2.putText(image, str('HI'), (x_pos, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale= 1.0, color=(0, 255, 0), thickness=1 )

            if output_path != '': out.write(image)
            # if show:
            # out.write(image)
            # print(img.shape)

            # img[onethirdHeight:img_h, 0:img_w] = image[0:img_h, 0:img_w]
            # import boto3
            # client = boto3.client('textract')
            # imgfilename = r"./IMAGES/croppedlicenseplate.jpg"
            # # print("hello")
            # # def get_file_from_filepath(filename):
            # with open(imgfilename, 'rb') as imgfile:
            #     imageBytes = bytearray(imgfile.read())
            #     # print(type(imageBytes))
            #
            # # imgbyte = get_file_from_filepath(imgfilename)
            #
            # result = client.detect_document_text(Document={'Bytes': imageBytes})
            # shri = result['Blocks']
            #
            # text_arr = []
            #
            # for item in shri:
            #     if item["BlockType"] == "LINE":
            #         text_arr.append(item['Text'])
            #     # else:
            #     #     print("Text is not clear")
            #
            # print(text_arr)
            # cv2.putText(image, str(text_arr), (50, 200),cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale= 1.0, color=(0, 255, 0), thickness=1 )

            ########################################################################################################################
            #image_path = image
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
            #
            # #image = cv2.imread(image)
            # #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #
            # reader = easyocr.Reader(['en', 'en'])
            # result = reader.readtext(image)
            # print(result[-1][-2])

            #image = cv2.putText(image, str(result), "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            ########################################################################################################################

            cv2.imshow('output', image)
            # cv2.imshow('output', img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()


# detect from webcam
def detect_realtime(Yolo, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3,
                    iou_threshold=0.45, rectangle_colors=''):
    times = []
    vid = cv2.VideoCapture(0)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4

    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        times.append(t2 - t1)
        times = times[-20:]

        ms = sum(times) / len(times) * 1000
        fps = 1000 / ms

        print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

        frame = draw_bbox(original_frame, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_frame, bboxes, read_class_names(CLASSES))
        image = cv2.putText(frame, "Time: {:.1f}FPS".format(fps), (0, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if output_path != '': out.write(frame)
        if show:
            cv2.imshow('output', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
