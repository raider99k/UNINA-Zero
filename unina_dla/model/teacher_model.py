from ultralytics import YOLO

def get_teacher_model(weights_path=None):
    """
    Initialize and return the YOLOv10-X teacher model.
    If weights_path is provided, loads the weights.
    Otherwise, loads the pretrained yolov10x.pt.
    """
    try:
        if weights_path:
            model = YOLO(weights_path)
        else:
            # Load standard YOLOv10-X pretrained on COCO
            # This will download the weights if not present
            model = YOLO('yolov10x.pt') 
            
        print(f"Teacher model loaded: {model.info()}")
        return model
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        return None

if __name__ == "__main__":
    # Smoke test
    get_teacher_model()
