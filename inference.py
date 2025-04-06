import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIG ---
class_names = ['Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle', 'Centipede', 
               'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox', 
               'Frog', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster', 'Harbor seal', 'Hedgehog', 'Hippopotamus', 
               'Horse', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 
               'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse', 'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 
               'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit', 'Raccoon', 'Raven', 'Red panda', 'Rhinoceros', 
               'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp', 'Snail', 'Snake', 
               'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Swan', 'Tick', 'Tiger', 'Tortoise', 'Turkey', 
               'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra']

model_path = 'best_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = models.resnet152(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Common transforms ---
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Function: Single image classification ---
def classify_single_image(model, image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = image_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, 5)

        predicted_idx = top_indices[0][0].item()
        predicted_class = class_names[predicted_idx]
        confidence = top_probs[0][0].item()

        # Plot image and predictions
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')

        top_classes = [class_names[idx] for idx in top_indices[0]]
        top_scores = top_probs[0].cpu().numpy() * 100
        y_pos = range(len(top_classes))

        plt.subplot(1, 2, 2)
        plt.barh(y_pos, top_scores, align='center')
        plt.yticks(y_pos, top_classes)
        plt.xlabel('Confidence (%)')
        plt.title('Top 5 Predictions')
        plt.tight_layout()
        plt.show()

        print(f"\nPrediction: {predicted_class} ({confidence * 100:.2f}%)")
        for i in range(5):
            print(f"{i+1}. {top_classes[i]}: {top_scores[i]:.2f}%")

        return predicted_class, confidence

    except Exception as e:
        print(f"Error: {e}")
        return None, 0.0

# --- Function: Evaluate entire test directory ---
def evaluate_test_directory(model, test_dir):
    correct, total = 0, 0
    predictions = {}
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)

    for true_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir): continue

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = image_transforms(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    _, pred_idx = torch.max(output, 1)
                    pred_idx = pred_idx.item()
                    predicted_class = class_names[pred_idx]

                predictions[img_path] = predicted_class
                total += 1
                if predicted_class == class_name:
                    correct += 1

                confusion_matrix[true_idx][pred_idx] += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    accuracy = correct / total if total else 0
    print(f"\nTest Set Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, cname in enumerate(class_names):
        total_i = np.sum(confusion_matrix[i])
        if total_i > 0:
            acc_i = confusion_matrix[i][i] / total_i
            print(f"{cname}: {acc_i:.4f} ({confusion_matrix[i][i]}/{total_i})")

    # Confusion Matrix Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    return accuracy, predictions

# --- RUN MODE ---
# Set this to either 'image' or 'folder'
run_mode = 'image'  # or 'folder'

if run_mode == 'image':
    image_path = '/kaggle/working/data/extracted/test/Bear/0df78ee76bafd3a9.jpg'
    classify_single_image(model, image_path)

elif run_mode == 'folder':
    test_dir = 'data/extracted/test'
    evaluate_test_directory(model, test_dir)
else:
    print("Invalid mode. Use 'image' or 'folder'.")

