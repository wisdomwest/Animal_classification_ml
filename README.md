# Animal Classification with ResNet-152

## Overview
This project implements an image classification pipeline using a pre-trained ResNet-152 model from PyTorch's `torchvision` library. It covers:

1. **Data Loading & Augmentation**: Loading training and testing datasets from directory structures, applying data transformations.
2. **Model Fine-Tuning**: Modifying the final fully connected layer of ResNet-152 to match the number of animal classes.
3. **Training Loop**: Training the model with learning rate scheduling and checkpointing.
4. **Evaluation & Inference**: Evaluating on a held-out test set with accuracy metrics, confusion matrix visualization, and single-image classification with top-5 predictions.

## Repository Structure
```
├── data/
│   └── extracted/
│       ├── train/        # Training images organized by class
│       └── test/         # Test images organized by class
├── best_model.pth        # Checkpoint of the best model
├── full_inference_and_eval.py  # Combined inference & evaluation script
├── inference.py          # (Optional) standalone inference script
└── train.py              # Training script (embedded at top of full script)
```

## Dependencies
- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow

Install via:
```bash
pip install torch torchvision numpy matplotlib pillow
```

## Data Preparation
1. Organize images in `data/extracted/train/<class_name>/` for training.
2. Organize images in `data/extracted/test/<class_name>/` for testing.
3. Ensure each class directory contains only image files (`.jpg`, `.jpeg`, `.png`).

## Training Script
The training code is located at the top of `full_inference_and_eval.py` (or in a separate `train.py`):

1. **Data Transforms**:
   - **Training**: Random resized crop to 224×224, random horizontal flip, normalization.
   - **Testing**: Resize to 256, center crop to 224, normalization.

2. **Model Setup**:
   - Load pre-trained ResNet-152 (`pretrained=True`).
   - Replace the final `fc` layer with a new `nn.Linear` matching `num_classes`.
   - Move model to `device` (GPU if available).

3. **Optimizer & Scheduler**:
   - **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate 0.001 and momentum 0.9.
   - **Scheduler**: StepLR decreases LR by a factor of 0.1 every 7 epochs.

4. **Training Loop**:
   ```python
   for epoch in range(num_epochs):
       for phase in ['train', 'test']:
           model.train() if phase=='train' else model.eval()
           for inputs, labels in dataloaders[phase]:
               outputs = model(inputs)
               loss = criterion(outputs, labels)
               if phase=='train':
                   loss.backward(); optimizer.step()
           if phase=='train': scheduler.step()
           # Track loss & accuracy
           # Save best model based on test accuracy
   ```

5. **Checkpointing**:
   - Saves `best_model.pth` whenever validation accuracy improves.

## Tricks & Optimizations
- **Transfer Learning**: Leveraging a ResNet-152 pretrained on ImageNet accelerates convergence and improves generalization on limited data.
- **Data Augmentation**: Random crops and flips increase data variability, reducing overfitting.
- **Learning Rate Scheduling**: StepLR reduces the learning rate mid-training to fine-tune weights and stabilize convergence.
- **Mixed-Precision (Optional)**: Integrate PyTorch's `torch.cuda.amp` for faster training and reduced memory usage.
- **Checkpointing Best Model**: Ensures the best-performing model on validation is preserved.
- **Batch Size**: A moderate batch size (32) balances GPU memory usage and gradient stability.

## Evaluation & Inference
### Evaluate Entire Test Set
```bash
# In full_inference_and_eval.py set run_mode='folder' add path to test_dir under data/extracted
python inference.py
```
- Computes overall accuracy and per-class accuracy.
- Plots a confusion matrix for error analysis.

### Classify a Single Image
```bash
# In full_inference_and_eval.py set run_mode='image' add path to image under data/extracted/test_dir
python full_inference_and_eval.py
```
- Displays the input image.
- Shows a bar chart of top-5 class probabilities.

## Extending & Customizing
- **Command-Line Interface**: Use `argparse` to pass `--mode`, `--image_path`, and `--test_dir` arguments.
- **Additional Augmentations**: Add color jitter, random rotation, or cutout for robustness.
- **Alternative Optimizers**: Try AdamW or Ranger for faster convergence.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and schedulers (e.g., CosineAnnealingLR).
- **Mixed-Precision**: Enable `torch.cuda.amp.autocast` for performance gains on NVIDIA GPUs.

