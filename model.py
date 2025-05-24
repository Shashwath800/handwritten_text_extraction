import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import os


class AdvancedOCRModel:
    def __init__(self, img_width=128, img_height=32, max_text_length=32):
        self.img_width = img_width
        self.img_height = img_height
        self.max_text_length = max_text_length

        # Character set (digits + uppercase + lowercase + space)
        self.characters = string.digits + string.ascii_letters + ' '
        self.char_to_num = {char: idx for idx, char in enumerate(self.characters)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.characters)}
        self.num_classes = len(self.characters)

        self.model = None
        self.prediction_model = None

    def build_crnn_model(self):
        """Build CRNN (CNN + RNN) model for text recognition"""

        # Input layers
        input_img = layers.Input(shape=(self.img_height, self.img_width, 1), name='image')
        labels = layers.Input(name='labels', shape=(None,), dtype='int32')
        input_length = layers.Input(name='input_length', shape=(1,), dtype='int32')
        label_length = layers.Input(name='label_length', shape=(1,), dtype='int32')

        # CNN layers for feature extraction
        conv_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_1)

        conv_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

        conv_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
        batch_norm_1 = layers.BatchNormalization()(conv_3)

        conv_4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(batch_norm_1)
        pool_4 = layers.MaxPooling2D(pool_size=(2, 1))(conv_4)

        conv_5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
        batch_norm_2 = layers.BatchNormalization()(conv_5)

        conv_6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_2)
        pool_6 = layers.MaxPooling2D(pool_size=(2, 1))(conv_6)

        conv_7 = layers.Conv2D(512, (2, 2), activation='relu')(pool_6)

        # Reshape for RNN layers
        squeezed = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(conv_7)

        # Bidirectional LSTM layers
        blstm_1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(squeezed)
        blstm_2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(blstm_1)

        # Dense layer
        outputs = layers.Dense(self.num_classes + 1, activation='softmax', name='dense')(blstm_2)

        # CTC loss layer
        loss_out = layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [outputs, labels, input_length, label_length]
        )

        # Create model
        self.model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)

        # Create prediction model (without CTC loss for inference)
        self.prediction_model = Model(inputs=input_img, outputs=outputs)

        return self.model

    def build_attention_model(self):
        """Build an attention-based OCR model"""

        # Encoder (CNN)
        input_img = layers.Input(shape=(self.img_height, self.img_width, 1))

        # Feature extraction
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

        # Reshape for attention mechanism
        batch_size = tf.shape(x)[0]
        feature_height = tf.shape(x)[1]
        feature_width = tf.shape(x)[2]
        feature_channels = x.shape[-1]

        # Flatten spatial dimensions but keep sequence dimension
        encoder_features = layers.Reshape((-1, feature_channels))(x)

        # Decoder with attention
        decoder_input = layers.Input(shape=(self.max_text_length,))
        decoder_embedding = layers.Embedding(self.num_classes + 1, 256)(decoder_input)

        # LSTM decoder
        decoder_lstm = layers.LSTM(512, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding)

        # Attention mechanism
        attention = layers.Attention()([decoder_outputs, encoder_features])

        # Concatenate attention with decoder outputs
        decoder_concat = layers.Concatenate()([decoder_outputs, attention])

        # Final dense layer
        decoder_dense = layers.Dense(self.num_classes + 1, activation='softmax')
        output = decoder_dense(decoder_concat)

        self.model = Model(inputs=[input_img, decoder_input], outputs=output)
        return self.model

    def ctc_lambda_func(self, args):
        """Lambda function for CTC loss"""
        y_pred, labels, input_length, label_length = args

        # CTC loss
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def ctc_loss_func(self, y_true, y_pred):
        """CTC loss function for compilation"""
        return y_pred

    def preprocess_image(self, image_path):
        """Preprocess image for OCR"""
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path

        if image is None:
            raise ValueError("Could not load image")

        # Resize image
        image = cv2.resize(image, (self.img_width, self.img_height))

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=[0, -1])

        return image

    def encode_text(self, text):
        """Encode text to numerical representation"""
        encoded = []
        for char in text:
            if char in self.char_to_num:
                encoded.append(self.char_to_num[char])
            else:
                encoded.append(self.char_to_num[' '])  # Default to space for unknown chars

        return np.array(encoded, dtype=np.int32)

    def decode_predictions(self, predictions):
        """Decode model predictions to text"""
        decoded_texts = []

        for prediction in predictions:
            # Get the most likely character at each time step
            decoded_chars = np.argmax(prediction, axis=-1)

            # Convert to text
            text = ""
            prev_char = -1

            for char_idx in decoded_chars:
                # Skip blank tokens (assuming blank token is at index num_classes)
                if char_idx != self.num_classes and char_idx != prev_char:
                    if char_idx < len(self.num_to_char):
                        text += self.num_to_char[char_idx]
                prev_char = char_idx

            decoded_texts.append(text.strip())

        return decoded_texts

    def compile_model(self, model_type='crnn'):
        """Compile the model with appropriate loss and metrics"""
        if model_type == 'crnn':
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=self.ctc_loss_func,
                metrics=['accuracy']
            )
        else:  # attention model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

    def train_model(self, train_images, train_labels, validation_images, validation_labels,
                    epochs=100, batch_size=32):
        """Train the OCR model"""

        # Prepare training data for CTC
        def prepare_ctc_data(images, labels):
            # Calculate input lengths based on actual CNN output width
            # After all pooling operations: width/4/2/2 = width/16 for height, width/4 for width
            input_lengths = np.ones(len(images)) * 31  # Fixed based on CNN architecture

            # Calculate label lengths and filter out labels that are too long
            valid_indices = []
            valid_labels = []
            valid_images = []

            for i, label in enumerate(labels):
                if len(label) <= 25:  # Keep some margin below input_length
                    valid_indices.append(i)
                    valid_labels.append(label)
                    valid_images.append(images[i])

            if len(valid_labels) == 0:
                raise ValueError("All labels are too long for the current model architecture")

            print(f"Filtered {len(labels) - len(valid_labels)} samples with labels too long")

            # Calculate label lengths for valid labels
            label_lengths = np.array([len(label) for label in valid_labels])

            # Pad labels
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            labels_padded = pad_sequences(valid_labels, padding='post', value=0)

            # Create dummy targets (CTC loss is computed in the model)
            dummy_targets = np.zeros((len(valid_images), 1))

            return [np.array(valid_images), labels_padded, input_lengths[:len(valid_images)],
                    label_lengths], dummy_targets

        # Prepare data
        train_inputs, train_targets = prepare_ctc_data(train_images, train_labels)
        val_inputs, val_targets = prepare_ctc_data(validation_images, validation_labels)

        # Prepare callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint('best_ocr_model.h5', save_best_only=True, monitor='val_loss')
        ]

        # Train the model
        history = self.model.fit(
            train_inputs, train_targets,
            validation_data=(val_inputs, val_targets),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict_text(self, image_path):
        """Predict text from image"""
        if self.prediction_model is None:
            raise ValueError("Model not built or loaded")

        # Preprocess image
        processed_image = self.preprocess_image(image_path)

        # Make prediction using the prediction model
        predictions = self.prediction_model.predict(processed_image)

        # Decode prediction
        decoded_text = self.decode_predictions(predictions)

        return decoded_text[0]

    def visualize_predictions(self, image_paths, predictions):
        """Visualize predictions on images"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, (img_path, pred_text) in enumerate(zip(image_paths[:6], predictions[:6])):
            if isinstance(img_path, str):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = img_path

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Predicted: {pred_text}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")

    def load_model(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'ctc_loss_func': self.ctc_loss_func}
        )
        print(f"Model loaded from {filepath}")

    def evaluate_model(self, test_images, test_labels):
        """Evaluate model performance"""
        if self.prediction_model is None:
            raise ValueError("Model not built or loaded")

        # Make predictions using prediction model
        predictions = self.prediction_model.predict(test_images)
        decoded_predictions = self.decode_predictions(predictions)

        # Calculate accuracy (simple character-level accuracy)
        correct_predictions = 0
        total_predictions = len(decoded_predictions)

        for pred, true_label in zip(decoded_predictions, test_labels):
            if isinstance(true_label, np.ndarray):
                # Convert numerical labels back to text
                true_text = ''.join(
                    [self.num_to_char.get(idx, '') for idx in true_label if idx < len(self.num_to_char)])
            else:
                true_text = true_label

            if pred.strip() == true_text.strip():
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Model Accuracy: {accuracy:.4f}")

        return accuracy, decoded_predictions


# Example usage and data preparation functions
def generate_synthetic_data(num_samples=1000, ocr_model=None):
    """Generate synthetic text images for training"""
    if ocr_model is None:
        ocr_model = AdvancedOCRModel()

    images = []
    labels = []

    # Simple synthetic data generation with shorter text
    for i in range(num_samples):
        # Create a blank image
        img = np.zeros((ocr_model.img_height, ocr_model.img_width), dtype=np.uint8)

        # Generate shorter random text (3-8 characters to stay well below input length limit)
        text_length = np.random.randint(3, 8)
        text = ''.join(np.random.choice(list(ocr_model.characters), text_length))

        # Simple text rendering (you'd use more sophisticated text-to-image generation)
        # This is just a placeholder - replace with actual text-to-image generation
        cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)

        images.append(img)
        labels.append(text)

    return np.array(images), labels


def main():
    """Main function to demonstrate OCR model usage"""

    # Initialize OCR model
    ocr = AdvancedOCRModel(img_width=128, img_height=32, max_text_length=32)

    # Build CRNN model
    model = ocr.build_crnn_model()
    print("CRNN Model Architecture:")
    model.summary()

    # Compile model
    ocr.compile_model('crnn')

    # Generate synthetic training data
    print("Generating synthetic training data...")
    train_images, train_labels = generate_synthetic_data(800, ocr)
    val_images, val_labels = generate_synthetic_data(200, ocr)

    # Preprocess data
    train_images = train_images.astype(np.float32) / 255.0
    train_images = np.expand_dims(train_images, axis=-1)

    val_images = val_images.astype(np.float32) / 255.0
    val_images = np.expand_dims(val_images, axis=-1)

    # Encode labels for CTC loss (don't pad here, let train_model handle it)
    train_labels_encoded = [ocr.encode_text(label) for label in train_labels]
    val_labels_encoded = [ocr.encode_text(label) for label in val_labels]

    # Train model
    print("Training model...")
    history = ocr.train_model(
        train_images, train_labels_encoded,
        val_images, val_labels_encoded,
        epochs=50, batch_size=16
    )

    # Evaluate model
    print("Evaluating model...")
    accuracy, predictions = ocr.evaluate_model(val_images, val_labels)

    # Save model
    ocr.save_model('advanced_ocr_model.h5')

    print("Training completed!")


if __name__ == "__main__":
    main()