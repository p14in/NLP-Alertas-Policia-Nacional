import os
import time
import torch
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from faker import Faker

# Configuración
MODEL_DIR = "models"
DATA_DIR = "datasets"
BATCH_SIZES = [1, 8, 16, 32, 64]
TEXT_SIZES = [100, 1000, 10000]

# Generación de datos simulados
def generate_fake_comments(n_samples=10000):
    fake = Faker("es_ES")
    return [fake.sentence(nb_words=np.random.randint(5, 20)) for _ in range(n_samples)]

# Función de evaluación de escalabilidad
def evaluar_escalabilidad_sentimientos(device_type="cpu", num_samples=10000):
    # Cargar modelo y tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "sentiment_model"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "sentiment_model"))
    
    # Configurar dispositivo
    device = torch.device(device_type)
    model.to(device)
    print(f"Probando con dispositivo: {device}")
    
    # Cargar o generar datos
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "teva_es_limpio.csv")).sample(num_samples)
        texts = df['text'].tolist()
    except FileNotFoundError:
        print("Archivo no encontrado. Generando datos simulados...")
        texts = generate_fake_comments(num_samples)
    
    dataset = Dataset.from_dict({"text": texts})
    results = []
    
    for batch_size in BATCH_SIZES:
        for text_size in TEXT_SIZES:
            print(f"\n🔹 Evaluando batch size: {batch_size}, text size: {text_size} ...")
            sample_texts = texts[:text_size]
            num_batches = len(sample_texts) // batch_size
            processing_times = []
            cpu_usage = []
            gpu_memory = []
            
            try:
                for i in tqdm(range(num_batches), desc=f"Procesando {batch_size}-batch"):
                    batch_texts = sample_texts[i * batch_size: (i + 1) * batch_size]
                    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model(**inputs)
                    end_time = time.time()
                    
                    processing_times.append(end_time - start_time)
                    cpu_usage.append(psutil.cpu_percent())
                    if device_type == "cuda":
                        gpu_memory.append(torch.cuda.memory_allocated() / (1024 ** 2))  # En MB
                        torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️ Memoria agotada con batch_size={batch_size}, text_size={text_size}. Saltando...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            avg_time = np.mean(processing_times) if processing_times else np.nan
            texts_per_second = text_size / sum(processing_times) if processing_times else np.nan
            avg_cpu = np.mean(cpu_usage) if cpu_usage else np.nan
            avg_gpu = np.mean(gpu_memory) if gpu_memory and device_type == "cuda" else "N/A"
            
            results.append({
                "batch_size": batch_size,
                "text_size": text_size,
                "texts_per_second": texts_per_second,
                "avg_time_per_batch": avg_time,
                "total_time": sum(processing_times) if processing_times else np.nan,
                "cpu_usage_percent": avg_cpu,
                "gpu_memory_mb": avg_gpu,
                "device": device_type
            })
            print(f"Batch size: {batch_size}, Text size: {text_size}, Textos/seg: {texts_per_second:.2f}, CPU: {avg_cpu:.2f}%, GPU Mem: {avg_gpu}")
    
    return pd.DataFrame(results)

# Función para graficar resultados
def plot_escalabilidad(results):
    fig, axes = plt.subplots(1, len(TEXT_SIZES), figsize=(15, 5), sharey=True)
    
    for idx, text_size in enumerate(TEXT_SIZES):
        ax = axes[idx]
        cpu_data = results[(results['device'] == 'cpu') & (results['text_size'] == text_size)]
        gpu_data = results[(results['device'] == 'cuda') & (results['text_size'] == text_size)]
        
        # Graficar textos por segundo
        ax.plot(cpu_data['batch_size'], cpu_data['texts_per_second'], 
                label='CPU', marker='o', linestyle='-', color='blue')
        if not gpu_data.empty:
            ax.plot(gpu_data['batch_size'], gpu_data['texts_per_second'], 
                    label='GPU', marker='s', linestyle='--', color='red')
        
        ax.set_xscale('log')
        ax.set_xlabel('Tamaño del lote')
        ax.set_title(f'Tamaño de texto: {text_size}')
        ax.grid(True, which="both", ls="--")
        ax.legend()
    
    axes[0].set_ylabel('Textos por segundo')
    plt.suptitle('Comparación de Escalabilidad: CPU vs GPU', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Gráfico adicional para uso de recursos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for text_size in TEXT_SIZES:
        cpu_data = results[(results['device'] == 'cpu') & (results['text_size'] == text_size)]
        gpu_data = results[(results['device'] == 'cuda') & (results['text_size'] == text_size)]
        
        # Uso de CPU
        ax1.plot(cpu_data['batch_size'], cpu_data['cpu_usage_percent'], 
                 label=f'CPU (Textos: {text_size})', marker='o', linestyle='-')
        if not gpu_data.empty:
            ax1.plot(gpu_data['batch_size'], gpu_data['cpu_usage_percent'], 
                     label=f'GPU (Textos: {text_size})', marker='s', linestyle='--')
        
        # Uso de memoria GPU (solo GPU)
        if not gpu_data.empty:
            ax2.plot(gpu_data['batch_size'], gpu_data['gpu_memory_mb'], 
                     label=f'GPU (Textos: {text_size})', marker='s', linestyle='--')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Tamaño del lote')
    ax1.set_ylabel('Uso de CPU (%)')
    ax1.set_title('Uso de CPU por dispositivo')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Tamaño del lote')
    ax2.set_ylabel('Memoria GPU (MB)')
    ax2.set_title('Uso de Memoria GPU')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Ejecutar
if __name__ == "__main__":
    # Probar con CPU
    cpu_results = evaluar_escalabilidad_sentimientos(device_type="cpu")
    
    # Probar con GPU si está disponible
    gpu_results = pd.DataFrame()
    if torch.cuda.is_available():
        gpu_results = evaluar_escalabilidad_sentimientos(device_type="cuda")
    
    # Combinar resultados
    results = pd.concat([cpu_results, gpu_results], ignore_index=True)
    
    # Guardar resultados
    results.to_csv("escalabilidad_sentimientos.csv", index=False)
    print("\n✅ Resultados guardados en 'escalabilidad_sentimientos.csv'")
    print(results)
    
    # Generar gráficos
    plot_escalabilidad(results)