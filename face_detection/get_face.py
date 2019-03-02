import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from face_detection.image_tensor import ImageTensor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("BASE_DIR", BASE_DIR)

model_file = os.path.join(BASE_DIR, "../tf_files/faces_retrained.pb")
label_file = os.path.join(BASE_DIR, "../tf_files/labels_retrained.txt")
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"
image_dir = "image_dir"


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def perform_detection():
    face_cascade = cv2.CascadeClassifier( os.path.join(BASE_DIR, '../cascades/haarcascade_frontalface_alt.xml'))

    graph = load_graph(model_file)
    labels = load_labels(label_file)

    cap = cv2.VideoCapture(0)
    with tf.Session(graph=graph) as sess:

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("gray", gray)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                # print(x,y,w,h)
                # roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
                roi_color = frame[y:y + h, x:x + w]

                tensor_image = ImageTensor.covert_image_to_tensor(roi_color)

                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: tensor_image})

                results = np.squeeze(results)

                top_k = results.argsort()[-5:][::-1]
                cur_res = "None"
                for i in top_k:
                    if results[i] > 0.85:
                        print(labels[i], results[i])
                        cur_res = str(labels[i])

                # img_item = "7.png"
                # cv2.imwrite(img_item, roi_color)

                color = (255, 0, 0)  # BGR 0-255
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

                cv2.putText(frame, cur_res, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


class FaceClassifier():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')

        self.graph = load_graph(model_file)
        self.labels = load_labels(label_file)
        self.input_name = "import/" + input_layer
        self.output_name = "import/" + output_layer

    def classify_image(self, frame):
        cur_results = {}
        with tf.Session(graph=self.graph) as sess:

            input_operation = self.graph.get_operation_by_name(self.input_name)
            output_operation = self.graph.get_operation_by_name(self.output_name)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("gray", gray)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                # print(x,y,w,h)
                # roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
                roi_color = frame[y:y + h, x:x + w]

                tensor_image = ImageTensor.covert_image_to_tensor(roi_color)

                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: tensor_image})

                results = np.squeeze(results)

                top_k = results.argsort()[-5:][::-1]
                # cur_res = "None"

                for i in top_k:
                    cur_results[self.labels[i]] = float(results[i])

        return cur_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--image_dir", help="Name of image directory to be processed")
    args = parser.parse_args()

    if args.image_dir:
        image_dir = args.image_dir
    if args.graph:
        model_file = args.graph
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    perform_detection()
