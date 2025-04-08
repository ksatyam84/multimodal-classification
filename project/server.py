import os
import io
import torch
from app import Flask, request, jsonify
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from model_builder import MultimodalGenreClassifier
from predict import create_text_transform, create_image_transform  # reuse our transform functions

app = Flask(__name__)

MODEL_PATH = 'multimodal_genre_classifier1.pth'

def load_checkpoint():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
        weights_only=False
    )
    return checkpoint

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    description = request.form.get('description')
    poster_file = request.files.get('poster')

    if not description:
        return jsonify({"error": "Movie description is required."}), 400

    # Load checkpoint and reconstruct transforms
    checkpoint = load_checkpoint()
    text_transform = create_text_transform(checkpoint['tokenizer_name'])
    image_transform = create_image_transform(checkpoint['image_transform_params'])
    
    # Reconstruct MultiLabelBinarizer using saved classes
    mlb = MultiLabelBinarizer()
    mlb.fit([checkpoint['mlb_classes']])
    
    # Initialize model and load state
    model = MultimodalGenreClassifier(num_genres=len(mlb.classes_))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Process text: create tokenized input for the model
    text = text_transform(description)
    
    # Process image: If a file is uploaded, load it; otherwise create a default black image.
    if poster_file:
        try:
            image = Image.open(poster_file).convert('RGB')
        except Exception as e:
            print("Error processing image, using default black image.", e)
            image = Image.new('RGB', (224, 224), color='black')
    else:
        image = Image.new('RGB', (224, 224), color='black')

    image = image_transform(image).unsqueeze(0)  # Add batch dimension
    
    # Get prediction from model
    with torch.no_grad():
        outputs = model(text, image)
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).float().cpu().numpy().reshape(1, -1)
    
    # Decode predictions
    genres = list(mlb.inverse_transform(predictions)[0])
    
    return jsonify({"genres": genres})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)