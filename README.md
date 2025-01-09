This repository contains a Flask-based web application that uses a trained deep learning model to classify garbage into various categories. The application allows users to upload images or capture images via a camera and predicts the type of garbage along with the confidence score.

Features

Image Upload: Upload an image from your device and classify it.

Camera Capture: Capture an image from your camera and classify it.

Real-Time Predictions: Predicts garbage category with a trained neural network.

Interactive UI: User-friendly interface for seamless interaction.

Garbage Categories

The application can classify garbage into the following categories:

Battery
Biological
Brown Glass
Cardboard
Cloth
Green Glass
Metal
Paper
Plastic
Shoe
White Glass


Requirements
Python Libraries
Flask
tensorflow
numpy
Pillow
base64


    .
    ├── app.py                 # Main Flask application
    ├── static/
    │   ├── uploaded_images/   # Stores uploaded images
    ├── templates/
    │   ├── index.html         # Homepage template
    │   ├── result.html        # Result page template
    ├── garbage_classification_model.h5  # Pretrained model
    ├── requirements.txt       # Required Python dependencies
    └── README.md              # Documentation


1. Clone the Repository

2. Set Up the Environment
Install the required Python packages:
Place the garbage_classification_model.h5 file in the root directory.

4. Run the Application
Start the Flask server:

python app.py

Access the app in your browser at http://127.0.0.1:5000/.

4. Upload or Capture an Image
Upload an image using the file upload option.
Alternatively, capture an image with your camera (ensure your browser supports camera access).
5. View the Result
The app will display the predicted garbage category along with the confidence percentage.

