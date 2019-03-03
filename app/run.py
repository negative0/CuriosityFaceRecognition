from flask import Flask, request, jsonify
from flask_restful import Api
import os
import cv2

# db_connect = create_engine('sqlite:///chinook.db')
from face_detection.get_face import FaceClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
api = Api(app)

classifier = FaceClassifier()


@app.route("/", methods=['GET'])
def rootTask():
    # language=HTML

    return '''
                <html lang="en">
                    <body>
                        <form action="/api/v1.0/task" method="post" enctype="multipart/form-data">
                            <input name="file" type="file">
                            <button type="submit">Submit</button>
                        </form>
                        <p>Multiple faces in one image also supported</p>
                        <p>Currently working for following faces:</p>
                        <ul>
                            <li>Shahrukh Khan</li>
                            <li>Priyanka Chopra</li>
                        </ul>
                    </body>
                </html>
            '''


@app.route("/api/v1.0/task", methods=['POST'])
def doTask():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({"Error": "File not present"})
    file = request.files['file']
    file_path = os.path.join(BASE_DIR, "upload", file.filename)
    file.save(file_path)
    image = cv2.imread(file_path)
    results = classifier.classify_image(image)
    return jsonify({"Ok": "File Uploaded", "result": results})


if __name__ == '__main__':
    app.run(port='8000', debug=True)
