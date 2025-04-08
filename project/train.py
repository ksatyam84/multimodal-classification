import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from data_setup import MovieDataset, create_datasets, custom_collate_fn
from engine import eval_step, train_step
from model_builder import MultimodalGenreClassifier
import logging

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()
    ]
)

# Config
BATCH_SIZE = 8
EPOCHS = 6
LEARNING_RATE = 2e-5
DEVICE = "mps" if torch.mps.is_available() else "cpu"

# Verify MPS availability
print("\n--- Device Check ---")
print(f"MPS Available: {torch.mps.is_available()}")
print(f"Using Device: {DEVICE.upper()}")
print(f"PyTorch Version: {torch.__version__}")
print("--------------------\n")

# Transforms
text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def text_transform(text):
    encoding = text_tokenizer(
        text, 
        padding='max_length', 
        max_length=512,
        truncation=True, 
        return_tensors="pt"
    )
    return {k: v.squeeze(0) for k, v in encoding.items()}

if __name__ == "__main__":
    try:
        print("Starting script...")
        
        # Dataset and DataLoader (unchanged)
        train_dataset, test_dataset = create_datasets(
                                            "/Users/kumarsatyam/python/basicsofai/project/bigdataset/archive/filtered_movies.csv",
                                            text_transform,
                                            image_transform,
                                            test_size=0.2
                                        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        
        test_loader = DataLoader(
                                    test_dataset,
                                    batch_size=BATCH_SIZE,
                                    collate_fn=custom_collate_fn
                                )
        # Model 
        # It converts genre labels (e.g., ["Action", "Comedy"]) into binary vectors (e.g., [1, 0, 1, 0]).
        model = MultimodalGenreClassifier(num_genres=len(train_dataset.dataset.mlb.classes_)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Training loop with accuracy
        print(f"\n{'Epoch':<10} {'Train Loss':<12} {'Train Accuracy':<15}")
        print("-" * 40)
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_step(model,
                train_loader, 
                optimizer,
                loss_fn,
                DEVICE)
            test_loss, test_acc = eval_step(model, test_loader, loss_fn, DEVICE)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")
        
            # Save model (unchanged)
            torch.save({
                'model_state_dict': model.state_dict(),
                'mlb_classes': train_dataset.dataset.mlb.classes_.tolist(),
                'tokenizer_name': 'bert-base-uncased',
                'image_transform_params': {
                    'resize': (224, 224),
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }, 'multimodal_genre_classifier.pth')
        
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")
        raise