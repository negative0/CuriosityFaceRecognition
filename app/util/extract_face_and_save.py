import cv2
import os
from PIL import Image
import argparse

allowed_images = ('png', 'jpeg', 'jpg')

def convert_images(image_dir):
    count = 0

    face_cascade = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_alt2.xml')
    # dirname = os.path.dirname(image_dir)
    result_dir = image_dir + "/result"

    for infile in os.listdir(image_dir):
        if not infile.endswith(allowed_images):
            continue
        count += 1
        print(infile, count)

        file_name = image_dir + "/" + infile

        """Error Check in file"""
        try:
            im = Image.open(file_name)
            im.verify()
        except OSError:
            print("Invalid")
            continue

        frame = cv2.imread(file_name)
        cv2.imshow("cur", frame)
        cv2.waitKey(5)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            # roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y + h, x:x + w]
            cv2.imshow("face", roi_color)
            cv2.imwrite(os.path.join(result_dir, infile), roi_color)

    return count


if __name__ == "__main__":
    image_dir = "../../input_data/faces/priyanka-chopra"
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_dir", help="Name of image directory to be processed")
    # args = parser.parse_args()
    #
    # if args.image_dir:
    #     image_dir = args.image_dir

    if not os.path.isdir(image_dir + "/result"):
        os.makedirs(image_dir + "/result")

    convert_images(image_dir)

    cv2.destroyAllWindows()


