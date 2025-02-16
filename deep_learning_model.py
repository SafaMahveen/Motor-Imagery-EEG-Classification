import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

class DeepLearningModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None
        self.preprocess_data()
    
    def preprocess_data(self):
        data = pd.read_csv(self.file_path)
        label_encoder = LabelEncoder()
        data['Epoch_encoded'] = label_encoder.fit_transform(data['Epoch'])
        X = data.drop(columns=['Epoch', 'Epoch_encoded'])
        y = data['Epoch_encoded']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_resampled, y_resampled = resample(X_scaled, y, replace=True, n_samples=len(y) * 3, random_state=42)
        y_resampled_one_hot = to_categorical(y_resampled)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_resampled, y_resampled_one_hot, test_size=0.2, random_state=42)
    
    def build_vae(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(256, activation='relu')(inputs)
        z_mean = layers.Dense(128)(x)
        z_log_var = layers.Dense(128)(x)
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling)([z_mean, z_log_var])
        latent_inputs = layers.Input(shape=(128,))
        x = layers.Dense(256, activation='relu')(latent_inputs)
        outputs = layers.Dense(input_shape[0], activation='sigmoid')(x)
        encoder = models.Model(inputs, z)
        decoder = models.Model(latent_inputs, outputs)
        vae = models.Model(inputs, decoder(encoder(inputs)))
        vae.compile(optimizer='adam', loss='mse')
        self.vae, self.vae_encoder = vae, encoder
    
    def train_vae(self):
        early_stopping_vae = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        self.vae.fit(self.X_train, self.X_train, epochs=150, batch_size=10, validation_split=0.2, callbacks=[early_stopping_vae])
        self.X_train_encoded_vae = self.vae_encoder.predict(self.X_train)
        self.X_test_encoded_vae = self.vae_encoder.predict(self.X_test)
    
    def build_dae(self):
        inputs = layers.Input(shape=(self.X_train_encoded_vae.shape[1],))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        bottleneck = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(bottleneck)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.X_train_encoded_vae.shape[1], activation='sigmoid')(x)
        dae = models.Model(inputs, outputs)
        dae.compile(optimizer='adam', loss='mse')
        self.dae, self.dae_encoder = dae, models.Model(inputs, bottleneck)
    
    def train_dae(self):
        early_stopping_dae = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        self.dae.fit(self.X_train_encoded_vae, self.X_train_encoded_vae, epochs=150, batch_size=10, validation_split=0.2, callbacks=[early_stopping_dae])
        self.X_train_encoded = self.dae_encoder.predict(self.X_train_encoded_vae).reshape((-1, 64, 1))
        self.X_test_encoded = self.dae_encoder.predict(self.X_test_encoded_vae).reshape((-1, 64, 1))
    
    def build_cnn_lstm(self):
        inputs = layers.Input(shape=(64, 1))
        x = layers.Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(128)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.y_train.shape[1], activation='softmax')(x)
        self.model = models.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train_cnn_lstm(self):
        self.build_cnn_lstm()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_cnn_lstm.keras', save_best_only=True, monitor='val_accuracy')
        history=self.model.fit(self.X_train_encoded, self.y_train, epochs=150, batch_size=10, validation_split=0.2, callbacks=[early_stopping, checkpoint])
        return history
    
    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.X_test_encoded, self.y_test)
        print(f'Test Accuracy: {test_accuracy:.2f}')
        y_pred = self.model.predict(self.X_test_encoded)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        print(confusion_matrix(y_true, y_pred_classes))
        print(classification_report(y_true, y_pred_classes))
    