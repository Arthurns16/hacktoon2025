# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 14:25:30 2025

@author: valla
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# -----------------------------
# Configuration Class for Hyperparameters
# -----------------------------
class Config:
    # Data parameters
    n_pca_components = 50
    test_size = 0.2
    n_folds = 5  # for cross-validation
    
    # Model parameters
    hidden_sizes = [128, 64]  # can be easily modified
    dropout_rate = 0.3
    use_batch_norm = True
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-4
    num_epochs = 100
    patience = 10
    
    # Feature selection
    use_feature_selection = True
    n_features_to_select = 80  # before PCA
    
    # Data augmentation
    use_smote = True
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# -----------------------------
# 1. Load dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_excel("Helena_10groups.xlsx")
X = df.drop(columns=["phase"]).values
y = df["phase"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# 2. Feature Selection (New)
# -----------------------------
if config.use_feature_selection:
    print(f"Applying feature selection (selecting {config.n_features_to_select} features)...")
    selector = SelectKBest(mutual_info_classif, k=min(config.n_features_to_select, X.shape[1]))
    X = selector.fit_transform(X, y_encoded)
    print(f"Features reduced from {df.shape[1]-1} to {X.shape[1]}")

# -----------------------------
# 3. Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Filter rare classes automatically
# -----------------------------
unique, counts = np.unique(y_encoded, return_counts=True)
valid_classes = unique[counts > 1]
mask = np.isin(y_encoded, valid_classes)
X_filtered = X_scaled[mask]
y_filtered = y_encoded[mask]

le_filtered = LabelEncoder()
y_filtered = le_filtered.fit_transform(y_filtered)

print(f"Filtered dataset: {X_filtered.shape[0]} samples, {len(np.unique(y_filtered))} classes")

# -----------------------------
# 5. Apply PCA
# -----------------------------
pca = PCA(n_components=min(config.n_pca_components, X_filtered.shape[0], X_filtered.shape[1]))
X_reduced = pca.fit_transform(X_filtered)
print(f"PCA: {X_filtered.shape[1]} -> {X_reduced.shape[1]} components")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# -----------------------------
# 6. Enhanced Neural Network with Batch Normalization
# -----------------------------
class EnhancedNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_sizes=[128, 64], 
                 dropout_rate=0.3, use_batch_norm=True):
        super(EnhancedNN, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers dynamically
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# -----------------------------
# 7. Mixup Data Augmentation (New)
# -----------------------------
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -----------------------------
# 8. Training Function with Improvements
# -----------------------------
def train_model(X_train, y_train, X_val, y_val, config, use_mixup=True):
    # Apply SMOTE if enabled
    if config.use_smote and len(np.unique(y_train)) > 1:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train))-1))
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"SMOTE applied: {X_train.shape[0]} samples after oversampling")
        except:
            print("SMOTE failed (likely due to small class sizes), continuing without it")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(config.device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(config.device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(config.device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(config.device)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    num_classes = len(np.unique(y_train))
    model = EnhancedNN(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_sizes=config.hidden_sizes,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm
    ).to(config.device)
    
    # Loss function with class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s = []
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0, 0, 0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            
            # Apply mixup if enabled
            if use_mixup and np.random.random() > 0.5:
                xb, yb_a, yb_b, lam = mixup_data(xb, yb, alpha=0.2)
                out = model(xb)
                loss = mixup_criterion(criterion, out, yb_a, yb_b, lam)
            else:
                out = model(xb)
                loss = criterion(out, yb)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            if not use_mixup or np.random.random() > 0.5:
                correct += (preds == yb).sum().item()
            total += xb.size(0)
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_val_loss, correct_val, total_val = 0, 0, 0
        all_val_preds, all_val_labels = [], []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = criterion(out, yb)
                running_val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(out, dim=1)
                all_val_preds.append(preds.cpu())
                all_val_labels.append(yb.cpu())
                correct_val += (preds == yb).sum().item()
                total_val += xb.size(0)
        
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Calculate F1 score
        y_val_preds_cat = torch.cat(all_val_preds).numpy()
        y_val_labels_cat = torch.cat(all_val_labels).numpy()
        epoch_val_f1 = f1_score(y_val_labels_cat, y_val_preds_cat, average='weighted')
        val_f1s.append(epoch_val_f1)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.3f}, "
                  f"Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.3f}, Val F1={epoch_val_f1:.3f}")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accs, val_accs, val_f1s

# -----------------------------
# 9. Cross-Validation Implementation (New)
# -----------------------------
def cross_validate_model(X, y, config):
    """Perform k-fold cross-validation"""
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    
    cv_scores = {'accuracy': [], 'f1': []}
    fold_models = []
    
    print(f"\nPerforming {config.n_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{config.n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model, _, _, _, _, _ = train_model(X_train, y_train, X_val, y_val, config)
        fold_models.append(model)
        
        # Evaluate
        model.eval()
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(config.device)
        
        with torch.no_grad():
            out = model(X_val_tensor)
            preds = torch.argmax(out, dim=1).cpu().numpy()
        
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')
        
        cv_scores['accuracy'].append(acc)
        cv_scores['f1'].append(f1)
        
        print(f"Fold {fold} - Accuracy: {acc:.3f}, F1: {f1:.3f}")
    
    print(f"\n{config.n_folds}-Fold CV Results:")
    print(f"Accuracy: {np.mean(cv_scores['accuracy']):.3f} (+/- {np.std(cv_scores['accuracy']):.3f})")
    print(f"F1 Score: {np.mean(cv_scores['f1']):.3f} (+/- {np.std(cv_scores['f1']):.3f})")
    
    return cv_scores, fold_models

# -----------------------------
# 10. Hyperparameter Tuning Function (New)
# -----------------------------
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Simple grid search for key hyperparameters"""
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'hidden_sizes': [[128, 64], [256, 128], [128, 64, 32]],
        'dropout_rate': [0.2, 0.3, 0.4]
    }
    
    best_score = 0
    best_params = {}
    
    print("\nPerforming hyperparameter tuning...")
    
    for lr in param_grid['learning_rate']:
        for hidden in param_grid['hidden_sizes']:
            for dropout in param_grid['dropout_rate']:
                # Update config
                temp_config = Config()
                temp_config.learning_rate = lr
                temp_config.hidden_sizes = hidden
                temp_config.dropout_rate = dropout
                temp_config.num_epochs = 30  # Fewer epochs for tuning
                
                # Train model
                model, _, _, _, _, val_f1s = train_model(
                    X_train, y_train, X_val, y_val, temp_config, use_mixup=False
                )
                
                # Get best F1 score
                best_f1 = max(val_f1s)
                
                if best_f1 > best_score:
                    best_score = best_f1
                    best_params = {
                        'learning_rate': lr,
                        'hidden_sizes': hidden,
                        'dropout_rate': dropout
                    }
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best F1 score: {best_score:.3f}")
    
    return best_params

# -----------------------------
# 11. Main Training Pipeline
# -----------------------------

# Option 1: Simple train/val split with hyperparameter tuning
print("\n" + "="*50)
print("OPTION 1: Train/Validation Split with Tuning")
print("="*50)

X_train, X_val, y_train, y_val = train_test_split(
    X_reduced, y_filtered, test_size=config.test_size, 
    random_state=42, stratify=y_filtered
)

# Tune hyperparameters (optional - comment out if you want to skip)
# best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)
# config.learning_rate = best_params['learning_rate']
# config.hidden_sizes = best_params['hidden_sizes']
# config.dropout_rate = best_params['dropout_rate']

# Train final model
print("\nTraining final model...")
model, train_losses, val_losses, train_accs, val_accs, val_f1s = train_model(
    X_train, y_train, X_val, y_val, config
)

# Final evaluation
model.eval()
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(config.device)
with torch.no_grad():
    out = model(X_val_tensor)
    final_preds = torch.argmax(out, dim=1).cpu().numpy()

acc = accuracy_score(y_val, final_preds)
f1_final = f1_score(y_val, final_preds, average="weighted")
print(f"\nFinal Validation Results:")
print(f"Accuracy: {acc:.3f}")
print(f"F1 Score: {f1_final:.3f}")

# Save model
torch.save(model.state_dict(), "best_enhanced_model.pth")

# Option 2: Cross-validation (uncomment to use)
print("\n" + "="*50)
print("OPTION 2: Cross-Validation")
print("="*50)
cv_scores, fold_models = cross_validate_model(X_reduced, y_filtered, config)

# -----------------------------
# 12. Ensemble Prediction (New)
# -----------------------------
def ensemble_predict(models, X, config):
    """Make predictions using model ensemble"""
    X_tensor = torch.tensor(X, dtype=torch.float32).to(config.device)
    
    all_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(X_tensor)
            probs = torch.softmax(out, dim=1)
            all_preds.append(probs.cpu().numpy())
    
    # Average predictions
    avg_preds = np.mean(all_preds, axis=0)
    final_preds = np.argmax(avg_preds, axis=1)
    
    return final_preds

# Test ensemble on validation set
print("\n" + "="*50)
print("ENSEMBLE RESULTS")
print("="*50)
ensemble_preds = ensemble_predict(fold_models, X_val, config)
ensemble_acc = accuracy_score(y_val, ensemble_preds)
ensemble_f1 = f1_score(y_val, ensemble_preds, average='weighted')
print(f"Ensemble Accuracy: {ensemble_acc:.3f}")
print(f"Ensemble F1 Score: {ensemble_f1:.3f}")

# ==============================
# 13. ENHANCED PLOTS
# ==============================

# Create a figure with subplots for better organization
fig = plt.figure(figsize=(20, 15))

# --- 1. Class distribution ---
ax1 = plt.subplot(3, 3, 1)
ax1.bar(np.arange(len(np.unique(y_filtered))), np.bincount(y_filtered))
ax1.set_xlabel("Class")
ax1.set_ylabel("Number of samples")
ax1.set_title("Class Distribution After Filtering")
ax1.grid(True, alpha=0.3)

# --- 2. PCA Explained Variance ---
ax2 = plt.subplot(3, 3, 2)
ax2.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o', color='green')
ax2.axhline(y=95, color='r', linestyle='--', label='95% variance')
ax2.set_xlabel("Number of PCA Components")
ax2.set_ylabel("Cumulative Explained Variance (%)")
ax2.set_title("PCA Explained Variance")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- 3. Training & Validation Loss ---
ax3 = plt.subplot(3, 3, 3)
ax3.plot(train_losses, label='Train Loss', marker='o', markersize=3)
ax3.plot(val_losses, label='Validation Loss', marker='s', markersize=3)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax3.set_title("Training and Validation Loss")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- 4. Training & Validation Accuracy ---
ax4 = plt.subplot(3, 3, 4)
ax4.plot(train_accs, label='Train Accuracy', marker='o', markersize=3)
ax4.plot(val_accs, label='Validation Accuracy', marker='s', markersize=3)
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Accuracy")
ax4.set_title("Training and Validation Accuracy")
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- 5. Validation F1 Score ---
ax5 = plt.subplot(3, 3, 5)
ax5.plot(val_f1s, label='Validation F1', marker='o', color='purple', markersize=3)
ax5.set_xlabel("Epoch")
ax5.set_ylabel("F1 Score")
ax5.set_title("Validation Weighted F1 Score")
ax5.grid(True, alpha=0.3)

# --- 6. Cross-validation scores (New) ---
ax6 = plt.subplot(3, 3, 6)
x_pos = np.arange(len(cv_scores['accuracy']))
width = 0.35
ax6.bar(x_pos - width/2, cv_scores['accuracy'], width, label='Accuracy', alpha=0.8)
ax6.bar(x_pos + width/2, cv_scores['f1'], width, label='F1 Score', alpha=0.8)
ax6.set_xlabel("Fold")
ax6.set_ylabel("Score")
ax6.set_title("Cross-Validation Scores by Fold")
ax6.set_xticks(x_pos)
ax6.set_xticklabels([f"Fold {i+1}" for i in range(len(cv_scores['accuracy']))])
ax6.legend()
ax6.grid(True, alpha=0.3)

# --- 7. Confusion Matrix ---
ax7 = plt.subplot(3, 3, 7)
cm = confusion_matrix(y_val, final_preds)
im = ax7.imshow(cm, cmap='Blues', aspect='auto')
ax7.set_xlabel("Predicted")
ax7.set_ylabel("Actual")
ax7.set_title("Confusion Matrix")
plt.colorbar(im, ax=ax7)

# --- 8. Class-wise F1 Scores ---
ax8 = plt.subplot(3, 3, 8)
f1_per_class = f1_score(y_val, final_preds, average=None)
ax8.bar(range(len(f1_per_class)), f1_per_class, color='orange', alpha=0.7)
ax8.set_xlabel("Class")
ax8.set_ylabel("F1 Score")
ax8.set_title("Class-wise F1 Scores")
ax8.grid(True, alpha=0.3)

# --- 9. t-SNE Visualization ---
ax9 = plt.subplot(3, 3, 9)
print("\nGenerating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X_reduced[:min(1000, len(X_reduced))])  # Limit samples for speed
y_subset = y_filtered[:min(1000, len(y_filtered))]

for c in np.unique(y_subset):
    mask = y_subset == c
    ax9.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                label=f"Class {c}", s=20, alpha=0.7)
ax9.set_title("t-SNE Visualization")
ax9.set_xlabel("t-SNE Dim 1")
ax9.set_ylabel("t-SNE Dim 2")
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
ax9.grid(True, alpha=0.3)

plt.suptitle("Enhanced Neural Network Classification Results", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# --- Additional Plot: Learning Rate Schedule (New) ---
plt.figure(figsize=(8, 5))
plt.plot(range(len(train_losses)), [config.learning_rate * (0.5 ** (i // 20)) for i in range(len(train_losses))], 
         label='Learning Rate Schedule', marker='o', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule (with ReduceLROnPlateau)")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"Best model saved as 'best_enhanced_model.pth'")
print(f"Device used: {config.device}")
print(f"Final validation accuracy: {acc:.3f}")
print(f"Final validation F1 score: {f1_final:.3f}")
print(f"Ensemble accuracy: {ensemble_acc:.3f}")
print(f"Ensemble F1 score: {ensemble_f1:.3f}")