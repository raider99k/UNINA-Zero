import os
from ultralytics import YOLO
import argparse

def train_teacher(data_yaml, epochs=300, batch_size=32, project='unina_dla_teacher', name='yolov10x_fsg'):
    """
    Train the YOLOv10-X teacher model on the FSG dataset.
    """
    # Load pretrained YOLOv10-X model
    model = YOLO('yolov10x.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=data_yaml,
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
        
        # Augmentations (can be tuned in data.yaml but overridden here if needed)
        hsv_h=0.015,
        hsv_s=0.7, 
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0, 
        mixup=0.1
    )
    
    # Export the best model
    success = model.export(format='onnx', opset=13, simplify=True)
    print(f"Training completed. Export success: {success}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv10-X Teacher')
    parser.add_argument('--data', type=str, default='unina_dla/config/unina_dla_data.yaml', help='Path to dataset yaml')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    train_teacher(args.data, args.epochs, args.batch)
