import os
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

#Load the PyTorch model
model = torch.load('model1.pth')
#model.eval()

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for handling video uploads
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    if file and allowed_file(file.filename):
        # Read the video file and extract frames
        video_buffer = file.read()
        frames = extract_frames(video_buffer)
        
        predicted_classes = []
        # open video file
        cap = cv2.VideoCapture('tmp_video.mp4')

# set frame interval (in seconds)
        interval = 0.5

# set folder to save frames
        folder_name = 'frames2'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

# initialize variables
        sec = 0
        count = 0

        while True:
    # set position to current second
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

    # read frame
            success, frame = cap.read()

    # check if frame was successfully extracted
            if success:
        # save frame to folder
                filename = os.path.join(folder_name, 'frame_{:04d}.jpg'.format(count))
                cv2.imwrite(filename, frame)

        # increment count
                count += 1

        # increment seconds
                sec += interval
            else:
        # break out of loop if end of video is reached
                break

# release resources
        cap.release()
        cv2.destroyAllWindows()
        for filename in os.listdir(folder_name):
    # Check if the file is an image
        # Open the image using PIL
            image =os.path.join(folder_name, filename)
            image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imagetensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(bytes(imagetensor))
            predicted_class = output.argmax().item()
            predicted_classes.append(predicted_class)
        
        return jsonify({'classes': predicted_classes})
    else:
        return jsonify({'error': 'Invalid file type'})

# Define a function to check if a file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'flv'}

# #Define a function to extract frames from a video file
# def extract_frames(video_buffer):
#     cap = cv2.VideoCapture(cv2.CAP_ANY)
#     cap.open(video_buffer)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         _, buffer = cv2.imencode('.jpg', frame)
#         yield buffer.tobytes()
#     cap.release()

def extract_frames(video_buffer):
    # Write video buffer to temporary file
    tmp_file = 'tmp_video.mp4'
    with open(tmp_file, 'wb') as f:
        f.write(video_buffer)

    # Open the temporary file with OpenCV
    cap = cv2.VideoCapture(tmp_file)

    # Loop through frames and yield them as bytes
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield buffer.tobytes()

    # Release the video capture object and delete the temporary file
    cap.release()
    os.remove(tmp_file)



# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
