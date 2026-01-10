import os
from ultralytics import YOLO
import argparse

import yaml

def train_teacher(data_yaml, epochs=300, batch_size=32, project='unina_dla_teacher', name='yolov10x_fsg', data_root=None):
    """
    Train the YOLOv10-X teacher model on the FSG dataset.
    """
    # Load pretrained YOLOv10-X model
    model = YOLO('yolov10x.pt')  # load a pretrained model (recommended for training)

    # Handle dataset path override
    data_arg = data_yaml
    if data_root is not None:
        print(f"Overriding dataset path with: {data_root}")
        with open(data_yaml, 'r') as f:
            data_dict = yaml.safe_load(f)
        data_dict['path'] = data_root
        data_arg = data_dict # pass dictionary instead of path

    # Train the model
    results = model.train(
        data=data_arg,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=0,  # Use GPU 0
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW', 
        lr0=0.001,
        cos_lr=True, # Cosine annealing
        
        # Augmentations are now controlled by the data.yaml file to ensure consistency
        # and prevent "signal contradiction" between script and config.
    )

    print(f"Teacher training completed.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv10-X Teacher')
    parser.add_argument('--data', type=str, default='unina_dla/config/unina_dla_data.yaml', help='Path to dataset yaml')
    parser.add_argument('--data_path', type=str, default=None, help='Override dataset root path (optional)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    train_teacher(args.data, args.epochs, args.batch, data_root=args.data_path)
