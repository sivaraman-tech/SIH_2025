from ultralytics import YOLO
import random

# Load YOLOv8 model pretrained on COCO (contains "cow" class)
model = YOLO("yolov8n.pt")

def is_cow(img_path, conf_threshold=0.5):
    """
    Check if a cow is present in the given image using YOLOv8.
    Args:
        img_path (str): Path to the image file.
        conf_threshold (float): Minimum confidence to accept detection.
    Returns:
        bool: True if cow detected, False otherwise.
    """
    results = model(img_path)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])   # class index
            label = model.names[cls_id]  # class name
            conf = float(box.conf[0])    # confidence score

            if label == "cow" and conf >= conf_threshold:
                print(f"✅ Cow detected with confidence {conf:.2f}")
                return True

    print("❌ No cow detected")
    return False


def predict_cow(img_path, age, body_weight, parity, historical_milk_yield):
    """
    Dummy cattle prediction function (replace later with real ML pipeline).
    Args:
        img_path (str): Path to cow image.
        age (float): Age of cow.
        body_weight (float): Cow's body weight.
        parity (int): Number of calvings.
        historical_milk_yield (float): Past milk yield in liters.
    Returns:
        dict: Prediction results.
    """
    # For now, generate mock results
    longevity = random.uniform(8, 15)
    milk_productivity = random.uniform(4000, 12000)
    repro_efficiency = random.uniform(1.0, 4.0)
    elite_prob = random.uniform(0.3, 0.95)

    results = {
        "longevity": f"{longevity:.1f} years",
        "milk_productivity": f"{milk_productivity:.0f} L/year",
        "reproductive_efficiency": f"{repro_efficiency:.2f} calves/year",
        "elite_dam": "Yes" if elite_prob > 0.5 else "No",
        "elite_dam_probability": f"{elite_prob*100:.1f}%",
        "ATP_score": f"{random.randint(60, 99)} / 100"
    }

    return results
