import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import BertTokenizer
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from model_builder import MultimodalGenreClassifier

# Required for safe loading
from numpy._core.multiarray import _reconstruct
from torch.serialization import add_safe_globals
add_safe_globals([_reconstruct])

def create_text_transform(tokenizer_name):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    return lambda text: tokenizer(
        text,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

def  create_image_transform(params):
    return transforms.Compose([
        transforms.Resize(params['resize']),
        transforms.ToTensor(),
        transforms.Normalize(mean=params['mean'], std=params['std'])
    ])

def predict_genre(overview_text, poster_url, model_path='multimodal_genre_classifier.pth'):
    try:
        # Load model
        checkpoint = torch.load(
            model_path,
            map_location=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
            weights_only=False
        )
        
        # Reconstruct transforms
        text_transform = create_text_transform(checkpoint['tokenizer_name'])
        image_transform = create_image_transform(checkpoint['image_transform_params'])
        
        # Reconstruct MLB properly
        mlb = MultiLabelBinarizer()
        mlb.fit([checkpoint['mlb_classes']])  # Key fix: Proper initialization
        
        # Initialize model
        model = MultimodalGenreClassifier(num_genres=len(mlb.classes_))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Process text
        text = text_transform(overview_text)
        
        # Process image
        try:
            response = requests.get(poster_url, timeout=30)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        image = image_transform(image).unsqueeze(0)
        
        # Predict and reshape
        with torch.no_grad():
            outputs = model(text, image)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float().cpu().numpy().reshape(1, -1)  # Ensure 2D shape
        
        return list(mlb.inverse_transform(predictions)[0])
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return []

if __name__ == "__main__":
    # Test prediction
    genres = predict_genre(
        "An unscrupulous newspaper editor searches for headlines at any cost",
        "https://m.media-amazon.com/images/M/MV5BNDk2YzM3MzEtMDRiOC00ZTY0LTgwZTUtYjNhNzQ1YjlmOWU5XkEyXkFqcGc@.jpg"
    )
    print("Predicted Genres:", genres if genres else "No predictions")