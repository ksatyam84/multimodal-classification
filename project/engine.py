import torch
from tqdm import tqdm

def train_step(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for texts, images, labels in tqdm(dataloader):
        texts = {k: v.to(device) for k, v in texts.items()}
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(texts, images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += labels.numel()  # Total labels = batch_size * num_classes
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def eval_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for texts, images, labels in tqdm(dataloader):
            texts = {k: v.to(device) for k, v in texts.items()}
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(texts, images)
            loss = loss_fn(outputs, labels)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct = (preds == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy