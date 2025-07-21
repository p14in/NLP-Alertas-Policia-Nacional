import os
import random
import numpy as np
import pandas as pd
import torch
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoConfig, set_seed
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Verificar GPU
if not torch.cuda.is_available():
    print("Error: No se detecta una GPU. El entrenamiento requiere una GPU.")
    sys.exit(1)

# Configuración
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
DATASET_PATH = "datasets/teva_es_limpio.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 3e-5
PATIENCE_LR = 2
FACTOR_LR = 0.8
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 3

# Mapeo de etiquetas
LABELS_MAP = {0: "Negativo", 1: "Neutral", 2: "Positivo"}

# Fijar semillas
def set_determinism(seed=42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Cargar y preparar dataset
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)  # Asegurar que las etiquetas sean enteros (0, 1, 2)

print("Distribución original de clases:")
print(df["label"].value_counts())

# División en entrenamiento y validación
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Tokenización y dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

train_dataset = CustomDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LEN)
val_dataset = CustomDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LEN)

# WeightedRandomSampler
train_labels = train_df["label"].values
class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
print("Pesos de clases:", class_weights)
sample_weights = [class_weights[label] for label in train_labels]
sample_weights = torch.tensor(sample_weights, dtype=torch.float)
train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Cargar modelo
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
device = torch.device("cuda")
model.to(device)

# Ajustar dropout
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.3

# Pérdida, optimizador y scheduler
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=FACTOR_LR, patience=PATIENCE_LR, verbose=True
)

# Entrenamiento
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, scheduler, early_stopping_patience):
    best_val_loss = float("inf")
    epochs_without_improve = 0

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validación
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        val_progress = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Validation Loss: {avg_val_loss:.4f} - Accuracy: {val_accuracy:.4f}")
        print(classification_report(all_labels, all_preds, target_names=list(LABELS_MAP.values())))

        # Matriz de confusión
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(LABELS_MAP.values()), yticklabels=list(LABELS_MAP.values()))
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusión - Época {epoch + 1}')
        plt.savefig(os.path.join(MODEL_DIR, f"confusion_matrix_epoch_{epoch + 1}.png"))
        plt.close()

        # Scheduler y early stopping
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            model.save_pretrained(os.path.join(MODEL_DIR, "best_model"))
            tokenizer.save_pretrained(os.path.join(MODEL_DIR, "best_model"))
            print("✅ Modelo mejorado guardado.")
        else:
            epochs_without_improve += 1
            print(f"Sin mejora durante {epochs_without_improve} época(s).")
            if epochs_without_improve >= early_stopping_patience:
                print("Early stopping activado. Finalizando entrenamiento.")
                break

# Ejecutar
set_determinism(42)
train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, EPOCHS, scheduler, EARLY_STOPPING_PATIENCE)