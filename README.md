# Motor-Imagery-EEG-Classification
This project implements deep learning models (VAE, DAE, and CNN-LSTM) to classify motor imagery EEG signals into imagined movements: Left Hand, Right Hand, Left Foot, and Right Foot. The system processes raw EEG data, removes noise, extracts key features, and predicts movement using an end-to-end deep learning pipeline.
<br>
Author - Safa Mahveen
<br>
## Overview
Motor Imagery EEG Classification using Deep Learning Techniques is a tool that simplifies EEG signal classification with Python's Tkinter, Pandas, TensorFlow, and Matplotlib. Users can load, preprocess, and classify EEG datasets using various deep learning techniques, including Variational Autoencoder (VAE), Deep Autoencoder (DAE), and Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM), all through an intuitive interface.

## Features
- **Load Datasets**: Users can import motor imagery EEG datasets in CSV format for analysis.
- **Data Preprocessing**: Label encoding, standardization, and resampling for class balance.
- **Noise Removal**: VAE is used to remove noise from EEG signals.
- **Feature Extraction**: DAE extracts meaningful latent representations from EEG data.
- **Classification Model**: CNN-LSTM is used for movement classification.
- **Visualizations**: Displays loss, accuracy curves, confusion matrix, and classification reports.
- **Intuitive Interface**: Easy navigation through a simple Graphical User Interface.

## Requirements
- Python 3.x
- Tkinter
- Pandas
- TensorFlow
- Scikit-learn
- Matplotlib
- Pillow

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/SafaMahveen/Motor-Imagery-EEG-Classification.git
   cd motor-imagery-eeg-classification
   ```
2. **Install the required packages:**
   ```bash
   pip install pandas numpy tensorflow scikit-learn matplotlib pillow tkinter
   ```

## Usage
Run the application using:
```bash
python graphical_user_interface.py
```
Once the application is running, you can load a dataset and start classifying EEG movements!

## Functional Overview
- **Load Dataset**: Click the "Upload Dataset" button to select a CSV file with motor imagery EEG data.
- **Preprocess Data**: Standardizes and resamples the dataset.
- **Train VAE**: Performs noise removal.
- **Train DAE**: Extracts key features.
- **Train CNN-LSTM**: Classifies motor imagery movements.
- **Evaluate Model**: Displays accuracy, confusion matrix, and classification report.
- **Classify Movement**: Predicts movement labels and displays corresponding images.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.
