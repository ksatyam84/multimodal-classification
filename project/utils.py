# utils.py
import matplotlib.pyplot as plt

def save_plots(train_loss, val_loss=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()