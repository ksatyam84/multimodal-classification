import torch
from torch.utils.data import Dataset
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import logging
from sklearn.model_selection import train_test_split



logging.basicConfig(
    level=logging.ERROR,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler() 
    ]
)

retry_strategy = Retry(
    total=3,  
    backoff_factor=1, 
    status_forcelist=[500, 502, 503, 504]
)

# Create a session with retry strategy
session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

class MovieDataset(Dataset):
    def __init__(self, csv_file, text_transform=None, image_transform=None, max_text_length=256):
        try:
            # Read CSV with pipe delimiter
            self.data = pd.read_csv(csv_file, delimiter='|')
            
            # Clean and validate data
            self._clean_data()
            
            # Prepare genre labels
            self.genres = self.data['Genre'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
            self.mlb = MultiLabelBinarizer()
            self.labels = self.mlb.fit_transform(self.genres)
            
            # Store transforms
            self.text_transform = text_transform
            self.image_transform = image_transform
            self.max_text_length = max_text_length
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def _clean_data(self):
        """Clean and validate the dataset."""
        # Fill missing values
        self.data['Genre'] = self.data['Genre'].fillna('Unknown')
        self.data['Poster_Url'] = self.data['Poster_Url'].fillna('')  # Fill missing URLs
        
        # Convert numeric columns
        self.data['Popularity'] = pd.to_numeric(self.data['Popularity'], errors='coerce').fillna(0)
        self.data['Vote_Count'] = pd.to_numeric(self.data['Vote_Count'], errors='coerce').fillna(0).astype(int)
        self.data['Vote_Average'] = pd.to_numeric(self.data['Vote_Average'], errors='coerce').fillna(0.0)
        
        # Validate required columns
        required_columns = [
            'Release_Date', 'Title', 'Overview', 'Popularity',
            'Vote_Count', 'Vote_Average', 'Original_Language',
            'Genre', 'Poster_Url'
        ]
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            # Get text
            text = self.data.iloc[idx]['Overview']
            
            # Get image
            img_url = self.data.iloc[idx]['Poster_Url']
            if pd.isna(img_url) or not isinstance(img_url, str) or not img_url.startswith('http'):
                raise ValueError(f"Invalid poster URL: {img_url}")
            
            try:
                response = session.get(img_url, timeout=30)  # Use session with retry
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except requests.exceptions.RequestException as e:
                # Log the error
                logging.error(f"Failed to fetch image from URL: {img_url}. Error: {str(e)}")
                raise ValueError(f"Failed to fetch image from URL: {img_url}")
            
            # Apply image transform
            if self.image_transform:
                image = self.image_transform(image)
            else:
                # Default transform if none provided
                image = transforms.ToTensor()(image)
            
            # Get labels
            label = torch.FloatTensor(self.labels[idx])
            
            # Apply text transform
            if self.text_transform:
                text = self.text_transform(text)  # Returns a dictionary
            else:
                # Default text dictionary if no transform is provided
                text = {
                    'input_ids': torch.zeros(self.max_text_length, dtype=torch.long),
                    'attention_mask': torch.zeros(self.max_text_length, dtype=torch.long)
                }
                
            return text, image, label
            
        except Exception as e:
            # Log the error
            logging.error(f"Error processing row {idx}: {str(e)}")
            # Return a fallback sample
            return self._get_fallback_sample()

    def _get_fallback_sample(self):
        """Returns a fallback sample for error handling."""
        # Log the use of a fallback sample
        logging.warning("Using fallback sample due to an error.")
        
        # Fallback text
        text = "Fallback overview text"
        if self.text_transform:
            text = self.text_transform(text)  # Apply text transform to get dictionary
        else:
            # Default text dictionary if no transform is provided
            text = {
                'input_ids': torch.zeros(self.max_text_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_text_length, dtype=torch.long)
            }
        
        # Fallback image
        image = Image.new('RGB', (224, 224), color='black')
        image = transforms.ToTensor()(image)  # Convert to tensor
        
        # Fallback label
        label = torch.zeros(len(self.mlb.classes_))
        
        return text, image, label
    
def custom_collate_fn(batch):
    """
    Custom collate function to handle dictionary inputs for text.
    """
    texts = [item[0] for item in batch]
    images = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # Stack images and labels
    images = torch.stack(images)  # Ensure images are tensors
    labels = torch.stack(labels)
    
    # Handle text dictionary
    text_dict = {
        'input_ids': torch.stack([t['input_ids'] for t in texts]),
        'attention_mask': torch.stack([t['attention_mask'] for t in texts])
    }
    
    return text_dict, images, labels

def create_datasets(csv_file, text_transform, image_transform, test_size=0.2):
    full_dataset = MovieDataset(csv_file, text_transform, image_transform)
    
    # Split indices
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=test_size,
        random_state=42,
    )
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    return train_dataset, test_dataset
