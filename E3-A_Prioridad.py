import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Directorios
DATA_DIR = "datasets"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if device.type == "cuda":
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("Error: No se detecta una GPU.")
    exit(1)

# Dataset personalizado
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Cargar y preparar dataset
def preparar_dataset_prioridades():
    df = pd.read_csv(os.path.join(DATA_DIR, "crisis_limpio.csv"))
    df = df.dropna(subset=['text', 'class_label'])
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['class_label'])
    
    # Imprimir distribución
    print("Distribución de class_label:")
    print(df['class_label'].value_counts())
    print("Mapeo de etiquetas:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"{label}: {idx}")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return train_df, test_df, label_encoder

# Tokenización
def tokenizar_dataset(df, tokenizer):
    encodings = tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
    return encodings, df['label'].values

# Entrenamiento
def entrenar_modelo_prioridades():
    train_df, test_df, label_encoder = preparar_dataset_prioridades()
    
    # Cargar tokenizer y modelo
    model_name = "dccuchile/bert-base-spanish-wwm-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model.to(device)
    
    # Ajustar dropout
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.3
    
    # Tokenizar datasets
    train_encodings, train_labels = tokenizar_dataset(train_df, tokenizer)
    test_encodings, test_labels = tokenizar_dataset(test_df, tokenizer)
    
    # Crear datasets
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    
    # WeightedRandomSampler
    class_weights = compute_class_weight('balanced', classes=np.arange(len(label_encoder.classes_)), y=train_labels)
    print("Pesos de clase:", class_weights)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Optimizador y pérdida
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.001)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Entrenamiento
    best_val_loss = float('inf')
    patience = 3
    epochs_without_improve = 0
    
    for epoch in range(10):
        print(f"\n===== Epoch {epoch + 1}/10 =====")
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_loader, desc="Training")
        for batch in train_progress:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
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
        val_progress = tqdm(test_loader, desc="Validation")
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_progress.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_val_loss = total_val_loss / len(test_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Validation Loss: {avg_val_loss:.4f} - Accuracy: {val_accuracy:.4f}")
        print("Reporte de clasificación:")
        print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
        
        # Matriz de confusión
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusión - Época {epoch + 1}')
        plt.savefig(os.path.join(MODEL_DIR, f"confusion_matrix_epoch_{epoch + 1}.png"))
        plt.close()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            model.save_pretrained(os.path.join(MODEL_DIR, "priority_model"))
            tokenizer.save_pretrained(os.path.join(MODEL_DIR, "priority_model"))
            print("✅ Modelo mejorado guardado.")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print("Early stopping activado.")
                break

# Ejecutar
if __name__ == "__main__":
    entrenar_modelo_prioridades()