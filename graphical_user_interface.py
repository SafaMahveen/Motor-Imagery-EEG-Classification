import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog
import pandas as pd
import tensorflow as tf
import subprocess
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk 
from deep_learning_model import DeepLearningModel
import numpy as np
from tkinter import  ttk
from sklearn.metrics import confusion_matrix, classification_report

model = None  
labels = ["LEFTFOOT_IMAGINE", "RIGHTHAND_IMAGINE", "LEFTHAND_IMAGINE", "RIGHTFOOT_IMAGINE"]  


BUTTON_STYLE = {
    "font": ("Arial", 14, "bold"),
    "fg": "white",
    "bg": "#9e0031",  
    "activebackground":"#FF5733",  
    "activeforeground":"white",
    "relief": "ridge",
    "bd": 3,
    "width": 20,
    "height": 2,
    "cursor": "hand2",
}

#  Function to change button color on hover
def on_enter(e):
    e.widget.config(bg="#003566")  

def on_leave(e):
    e.widget.config(bg="#9e0031")  

#  Set Background Image
def set_background(image_path):
    img = Image.open(image_path)
    img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()))  
    bg_image = ImageTk.PhotoImage(img)
    bg_label = tk.Label(root, image=bg_image)
    bg_label.image = bg_image  
    bg_label.place(relwidth=1, relheight=1)

BG_IMAGE_PATH = r".\Motor-Imagery-EEG-Classification\images_for_gui\bg_image.png"
root = tk.Tk()
root.title("Deep Learning GUI")
root.geometry("900x700")  
set_background(BG_IMAGE_PATH)
status_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="white", bg="black")
status_label.place(relx=0.15, rely=0.25, anchor="center")  
status_label.lift()  

title_label = tk.Label(root, text="CLASSIFICATION OF MOTOR IMAGERY \nEEG SIGNALS\n USING\n DEEP LEARNING TECHNIQUES", font=("Arial", 20, "bold"), fg="white",bg="black")
title_label.pack(pady=40)
buttons = [
    ("Project Video", lambda: play_video()),
    ("Upload Dataset", lambda: upload_file()),
    
    ("Preprocess Data", lambda: preprocess_data()),
    ("Train VAE", lambda: train_vae()),
    ("Train DAE", lambda: train_dae()),
    ("Train CNN-LSTM", lambda: train_cnn_lstm()),
    ("Evaluate Model", lambda: evaluate_model()),
    ("Classify Movement", lambda: classify_movement()),
  
]
for text, command in buttons:
    btn = tk.Button(root, text=text, command=command, **BUTTON_STYLE)
    btn.pack(pady=8)  
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

def upload_file():
    global model
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        model = DeepLearningModel(file_path)  
        messagebox.showinfo("File Selected", f"File Loaded: {file_path}")


def play_video():
    video_path = r"E:\MAJOR_PROJECT\project_animated_video.mp4"
    subprocess.run(["start", "wmplayer", video_path], shell=True)


def preprocess_data():
    if model:
        data = pd.read_csv(model.file_path)
        
       
        if 'Epoch' in data.columns:
            raw_data = data.drop(columns=['Epoch'])
        else:
            raw_data = data  

        model.preprocess_data()
        visualize_comparison(raw_data.values, model.X_train, "Raw Data", "After Preprocessing")
    else:
        messagebox.showwarning("Warning", "Upload a file first!")


def train_vae():
    if model:
        status_label.config(text="Training VAE... Please wait ⏳")
        root.update()
        model.build_vae((model.X_train.shape[1],))
        model.train_vae()
        visualize_comparison(model.X_train, model.X_train_encoded_vae, "After Preprocessing", "After VAE Encoding")
        status_label.config(text="VAE Training Completed ✅")
        
    else:
        messagebox.showwarning("Warning", "Upload and preprocess data first!")

def train_dae():
    if model:
        status_label.config(text="Training DAE... Please wait ⏳")
        root.update()
        model.build_dae()
        model.train_dae()
        visualize_comparison(model.X_train_encoded_vae, model.X_train_encoded, "After VAE Encoding", "After DAE Encoding")
        status_label.config(text="DAE Training Completed ✅")
        
    else:
        messagebox.showwarning("Warning", "Train VAE first!")

def train_cnn_lstm():
    if model:
        status_label.config(text="Training CNN-LSTM... Please wait ⏳")
        root.update()
        history = model.train_cnn_lstm()  
        
       
        plt.figure(figsize=(12, 5))

        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epochs")
        plt.legend()

        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()
        status_label.config(text="CNN-LSTM Training Completed ✅")
        
    else:
        messagebox.showwarning("Warning", "Train DAE first!")


def evaluate_model():
    if model:
        test_loss, test_accuracy = model.model.evaluate(model.X_test_encoded, model.y_test)
        y_pred = model.model.predict(model.X_test_encoded)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(model.y_test, axis=1)
        
        report = classification_report(y_true, y_pred_classes)
        cm = confusion_matrix(y_true, y_pred_classes)
        
        result_window = tk.Toplevel(root)
        result_window.title("Model Evaluation Results")
        result_window.geometry("600x400")
        
        result_text = f"Test Accuracy: {test_accuracy:.2f}\n\nClassification Report:\n{report}\n\nConfusion Matrix:\n{cm}"
        
        text_area = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, width=70, height=20)
        text_area.insert(tk.INSERT, result_text)
        text_area.config(state=tk.DISABLED)
        text_area.pack(padx=10, pady=10)
    else:
        messagebox.showwarning("Warning", "Train the CNN-LSTM model first!")
MOVEMENT_IMAGES = {
    "LEFTFOOT_IMAGINE": r".\Motor-Imagery-EEG-Classification\images_for_gui\left_foot.png",
    "RIGHTHAND_IMAGINE": r".\Motor-Imagery-EEG-Classification\images_for_gui\right_hand.jpg",
    "LEFTHAND_IMAGINE": r".\Motor-Imagery-EEG-Classification\images_for_gui\left_hand.jpg",
    "RIGHTFOOT_IMAGINE": r".\Motor-Imagery-EEG-Classification\images_for_gui\right_foot.png"
}
MOVEMENT_NAMES = {
    "LEFTHAND_IMAGINE": "LEFT HAND",
    "RIGHTHAND_IMAGINE": "RIGHT HAND",
    "LEFTFOOT_IMAGINE": "LEFT FOOT",
    "RIGHTFOOT_IMAGINE": "RIGHT FOOT"
}

prediction_image_label = tk.Label(root, bg="black")
prediction_image_label.place(relx=0.15, rely=0.5, anchor="center")  


prediction_text_label = tk.Label(root, text="", font=("Arial", 12, "bold"), fg="white", bg="black")
prediction_text_label.place(relx=0.15, rely=0.63, anchor="center")  

def classify_movement():
    if model:
        try:
            choice = messagebox.askyesno("Input Method", "Do you want to enter values manually? (Yes = Manual, No = Select from Data)")

            if choice:  # Manual Input
                input_values = simpledialog.askstring("Classify Movement", "Enter feature values separated by commas:")
                if input_values:
                    input_array = np.array([float(val) for val in input_values.split(",")]).reshape(1, -1)
            else:  # Select from Dataset
                data = pd.read_csv(model.file_path)

                if 'Epoch' in data.columns:
                    X = data.drop(columns=['Epoch'])
                else:
                    X = data  

                row_idx = simpledialog.askinteger("Select Row", f"Enter row number (0 - {len(X)-1}):")
                if row_idx is not None and 0 <= row_idx < len(X):
                    input_array = X.iloc[row_idx].values.reshape(1, -1)
                else:
                    messagebox.showwarning("Warning", "Invalid row number!")
                    return

            
            input_array = input_array.reshape(1, input_array.shape[1], 1)

            input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)

            prediction = model.model.predict(input_tensor, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            predicted_label = labels[predicted_class]
            prediction_confidence = prediction[predicted_class] * 100

            formatted_label = MOVEMENT_NAMES.get(predicted_label, predicted_label)

            
            prediction_image_label.config(image=None)  

            image_path = MOVEMENT_IMAGES.get(predicted_label, None)
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                img = img.resize((250, 250), Image.LANCZOS)
                img = ImageTk.PhotoImage(img)

                prediction_image_label.config(image=img)
                prediction_image_label.image = img  
            else:
                messagebox.showwarning("Image Not Found", f"Image for {formatted_label} not found!")

            prediction_text_label.config(text=f"Classified Movement: {formatted_label}\nAccuracy: {prediction_confidence:.2f}%")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


def visualize_comparison(data_before, data_after, title_before, title_after):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if data_before is not None:
        plt.hist(data_before.flatten(), bins=50, alpha=0.7, color='b')
    plt.title(title_before)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(data_after.flatten(), bins=50, alpha=0.7, color='r')
    plt.title(title_after)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

root.mainloop()
