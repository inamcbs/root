from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO('/path/to/your/best.pt')  # Change to your model path

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_path = '/tmp/' + image.filename
    image.save(image_path)

    # Run inference
    results = model.predict(source=image_path, conf=0.01, task='segment')
    
    # Convert results to JSON format
    response = {
        'predictions': results.pandas().xyxy[0].to_dict(orient="records")
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
