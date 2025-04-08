# model_builder.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import BertModel, BertTokenizer

class MultimodalGenreClassifier(nn.Module):
    def __init__(self, num_genres, text_model_name='bert-base-uncased'):
        super().__init__()
       
        self.bert = BertModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 256)
        
        
        self.cnn = resnet50(pretrained=True)
        self.image_fc = nn.Linear(1000, 256)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_genres )
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