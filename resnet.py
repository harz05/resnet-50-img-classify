import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from PIL import Image

# Model Creation
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Gradient-CAM Implementation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Training Loop
def train_model(model, train_data, val_data, epochs=10):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[ 
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
    )
    return history

# Explanation Generator for multiple images (online URL)
def generate_explanation_for_multiple_images(img_urls, model):
    results = []
    
    for img_url in img_urls:
        # Fetch image from URL
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))  # Resize to 224x224 as required by ResNet50
        img_array = np.array(img)
        
        # Preprocess image array for ResNet50
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        # Make prediction
        preds = model.predict(img_array)
        heatmap = make_gradcam_heatmap(img_array, model, 'conv5_block3_out')
        
        # Generate explanation based on highlighted areas
        explanation = []
        if np.mean(heatmap) > 0.5:
            explanation.append("High attention to texture inconsistencies")
        if np.max(heatmap) > 0.8:
            explanation.append("Strong artifacts detected in specific regions")
        if np.std(heatmap) > 0.2:
            explanation.append("Unusual patterns in image structure")
        
        result = {
            'img_url': img_url,
            'prediction': 'AI-generated' if preds[0][0] > 0.5 else 'Real',
            'confidence': preds[0][0],
            'explanation': ", ".join(explanation),
            'heatmap': heatmap
        }
        
        results.append(result)
    
    return results

# Full pipeline usage example for multiple images from online URLs
def main():
    # Create and train model
    model = create_model()
    
    # List of image URLs to process
    img_urls = [
        "https://your-image-url-1.com/image.jpg",
        "https://your-image-url-2.com/image.jpg",
        "https://your-image-url-3.com/image.jpg"
    ]  # Replace with actual URLs
    
    # Generate explanation for multiple images
    results = generate_explanation_for_multiple_images(img_urls, model)
    
    # Output the results for each image
    for result in results:
        print(f"Image URL: {result['img_url']}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']})")
        print(f"Explanation: {result['explanation']}")
        print("="*50)

if __name__ == "__main__":
    main()
