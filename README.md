# Coin Classification Project


### 1. Prepare the Dataset
Run the following command to split the dataset into training, validation, and test subsets:
```bash
python get_data.py
```
This will create a `dataset` folder with `train`, `val`, and `test` subdirectories, based on the `coindataset` directory.

### 2. Train the Model
Train the Vision Transformer (ViT) model using the prepared dataset:
```bash
python train.py
```
This will:
- Train the model for the specified number of epochs.
- Save the trained model weights as `fine_tuned_model.pth`.
- Save training and validation metrics to `training_metrics.csv`.
- Generate a confusion matrix for test dataset.

### 3. Test the Model
Evaluate the trained model on the test dataset:
```bash
python test_model.py
```
This will output the test accuracy for a specific model.

### 4. Generate Accuracy and Loss Plots
Create plots for training and validation accuracy and loss across epochs:
```bash
python validationplot.py
```
This will generate:
- `train_val_loss_plot.png`: Training and validation loss over epochs.
- `train_val_acc_plot.png`: Training and validation accuracy over epochs.

## Project Files
- `get_data.py`: Splits the dataset into train/val/test subsets.
- `train.py`: Fine-tunes the ViT model and saves the training metrics.
- `test_model.py`: Evaluates the trained model on the test set.
- `validationplot.py`: Generates accuracy and loss plots.
- `engine.py`: Defining training and testing process.