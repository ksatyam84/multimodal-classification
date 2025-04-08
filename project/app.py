from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.preprocessing import MultiLabelBinarizer
import time

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model definition
class MultimodalGenreClassifier(nn.Module):
    def __init__(self, num_genres, text_model_name='bert-base-uncased'):
        super().__init__()
        # Text branch
        self.bert = BertModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 256)
        
        # Image branch
        self.cnn = resnet50(pretrained=True)
        self.image_fc = nn.Linear(1000, 256)
        
        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_genres)
        )
        
    def forward(self, text, image):
        # Text features
        text_output = self.bert(
            input_ids=text['input_ids'],
            attention_mask=text['attention_mask']
        )
        text_features = self.text_fc(text_output.last_hidden_state[:, 0, :])
        
        # Image features
        image_features = self.cnn(image)
        image_features = self.image_fc(image_features)
        
        # Concatenate features
        combined = torch.cat((text_features, image_features), dim=1)
        
        return self.classifier(combined)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_text_transform(tokenizer_name):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    return lambda text: tokenizer(
        text,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

def create_image_transform(params):
    return transforms.Compose([
        transforms.Resize(params['resize']),
        transforms.ToTensor(),
        transforms.Normalize(mean=params['mean'], std=params['std'])
    ])

def predict_genre_with_image(overview_text, image_file, model_path='multimodal_genre_classifier.pth'):
    try:
        # Load model
        checkpoint = torch.load(
            model_path,
            map_location=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
            weights_only=False
        )
        
       
        text_transform = create_text_transform(checkpoint['tokenizer_name'])
        image_transform = create_image_transform(checkpoint['image_transform_params'])
        
        # Reconstruct MLB
        mlb = MultiLabelBinarizer()
        mlb.fit([checkpoint['mlb_classes']])
        
       
        model = MultimodalGenreClassifier(num_genres=len(mlb.classes_))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
      
        text = text_transform(overview_text)
        
       
        try:
            image = Image.open(image_file).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        image = image_transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(text, image)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            # If no genre reaches threshold, choose the genre with maximum probability.
            if predictions.sum() == 0:
                max_idx = torch.argmax(probs, dim=1)
                predictions[0, max_idx] = 1.0
            predictions = predictions.cpu().numpy().reshape(1, -1)
        
        gnr = list(mlb.inverse_transform(predictions)[0])
        
        return gnr
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return []

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        overview_text = request.form.get('overview', '')
        model_path = 'multimodal_genre_classifier.pth'
        
        # Handle file upload
        if 'poster_image' not in request.files:
            return render_template('index.html', error='No file part', overview=overview_text)
        
        file = request.files['poster_image']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file', overview=overview_text)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to make filename unique
            timestamp = str(int(time.time()))
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Make prediction
            try:
                with open(file_path, 'rb') as image_file:
                    genres = predict_genre_with_image(overview_text, image_file, model_path)
                
                # Keep the file and return response
                return render_template('index.html',
                                    genres=genres,
                                    overview=overview_text,
                                    image_path=f'uploads/{unique_filename}',
                                    error=None)
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return render_template('index.html',
                                    genres=[],
                                    overview=overview_text,
                                    image_path=None,
                                    error=str(e))
        
        return render_template('index.html',
                            genres=[],
                            overview=overview_text,
                            image_path=None,
                            error='Invalid file format')
    
    return render_template('index.html', genres=None, overview='', image_path=None, error=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)