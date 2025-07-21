import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suprimir advertencias de TensorFlow
import pandas as pd
from datasets import load_dataset
import torch
from tqdm import tqdm
import re
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

# Directorio para almacenar datasets
DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

# Verificar disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if device.type == "cuda":
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("GPU no detectada, usando CPU. La traducción será más lenta.")

# Verificar sentencepiece
try:
    import sentencepiece
    print("SentencePiece instalado correctamente")
except ImportError:
    print("Error: SentencePiece no está instalado. Instálalo con: pip install sentencepiece")
    exit(1)

# Función para descargar datasets
def descargar_datasets():
    datasets = {
        "tweet_eval": {"path": "cardiffnlp/tweet_eval", "subset": "sentiment"},
        "crisis_dataset": {"path": "QCRI/CrisisBench-english", "subset": "humanitarian"}
    }
    for name, info in datasets.items():
        local_path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(local_path):
            print(f"Descargando {name} desde HuggingFace...")
            try:
                dataset = load_dataset(info["path"], info["subset"])
                df = dataset['train'].to_pandas()
                df.to_csv(local_path, index=False)
                print(f"{name} descargado y guardado en {local_path}")
            except Exception as e:
                print(f"Error al descargar {name}: {e}")
                raise e
        else:
            print(f"{name} ya existe en {local_path}")

# Función para limpiar texto
def limpiar_texto(texto):
    texto = str(texto)
    texto = re.sub(r'@[\w_]+', '', texto)  # Eliminar menciones
    texto = re.sub(r'#\w+', '', texto)  # Eliminar hashtags
    texto = re.sub(r'http\S+|www.\S+', '', texto)  # Eliminar URLs
    texto = re.sub(r'\s+', ' ', texto).strip()  # Normalizar espacios
    texto = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ¿?¡!.,:;_\-\(\)/\s]', '', texto)
    return texto if texto else None

# Función para traducir al español
def traducir_texto(textos, model, tokenizer, device, batch_size=16):
    try:
        batches = [textos[i:i + batch_size] for i in range(0, len(textos), batch_size)]
        traducciones = []
        for batch in tqdm(batches, desc="Traduciendo"):
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                translated = model.generate(**inputs)
            traducciones.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
        return traducciones
    except Exception as e:
        print(f"Error al traducir: {e}")
        return textos

# Función para sobremuestreo aleatorio
def sobremuestreo_aleatorio(df, label_column):
    # Calcular distribución de clases
    class_counts = df[label_column].value_counts()
    print(f"Distribución original de {label_column}:")
    print(class_counts)
    
    # Encontrar el número máximo de ejemplos
    max_count = class_counts.max()
    
    # Lista para almacenar los dataframes balanceados
    balanced_dfs = []
    
    # Iterar sobre cada clase
    for label in class_counts.index:
        class_df = df[df[label_column] == label]
        # Si la clase tiene menos ejemplos que max_count, sobremuestrear
        if len(class_df) < max_count:
            oversample_size = max_count - len(class_df)
            oversampled_rows = class_df.sample(n=oversample_size, replace=True, random_state=42)
            balanced_dfs.append(pd.concat([class_df, oversampled_rows]))
        else:
            balanced_dfs.append(class_df)
    
    # Combinar todos los dataframes
    balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
    
    # Imprimir nueva distribución
    print(f"\nDistribución después de sobremuestreo para {label_column}:")
    print(balanced_df[label_column].value_counts())
    
    return balanced_df

# Procesar datasets
def procesar_datasets():
    # Cargar modelo y tokenizador
    model_name = "Helsinki-NLP/opus-mt-en-es"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Procesar tweet_eval (sentimientos)
    teva_es_path = os.path.join(DATA_DIR, "teva_es_limpio.csv")
    if os.path.exists(teva_es_path):
        print(f"teva_es_limpio.csv ya existe en {teva_es_path}. Saltando procesamiento.")
        df_tweet_eval = pd.read_csv(teva_es_path)
        df_tweet_eval = df_tweet_eval.dropna(subset=['text', 'label'])
        # Aplicar sobremuestreo
        df_tweet_eval = sobremuestreo_aleatorio(df_tweet_eval, 'label')
        df_tweet_eval.to_csv(teva_es_path, index=False)
        print("teva_es_limpio.csv actualizado con clases balanceadas.")
    else:
        tweet_eval_path = os.path.join(DATA_DIR, "tweet_eval.csv")
        try:
            df_tweet_eval = pd.read_csv(tweet_eval_path)
        except FileNotFoundError:
            print("Error: tweet_eval.csv no encontrado.")
            return

        df_tweet_eval['text'] = df_tweet_eval['text'].apply(limpiar_texto)
        df_tweet_eval = df_tweet_eval.dropna(subset=['text', 'label'])

        print("Traduciendo tweet_eval al español...")
        df_tweet_eval['text'] = traducir_texto(df_tweet_eval['text'].tolist(), model, tokenizer, device)

        df_tweet_eval = sobremuestreo_aleatorio(df_tweet_eval, 'label')
        columnas_tweet_eval = ['text', 'label']
        df_tweet_eval = df_tweet_eval[columnas_tweet_eval]
        df_tweet_eval.to_csv(teva_es_path, index=False)
        print("tweet_eval limpio guardado como teva_es_limpio.csv")

    # Procesar crisis_dataset
    crisis_path = os.path.join(DATA_DIR, "crisis_dataset.csv")
    try:
        df_crisis = pd.read_csv(crisis_path)
    except FileNotFoundError:
        print("Error: crisis_dataset.csv no encontrado.")
        return

    df_crisis['text'] = df_crisis['text'].apply(limpiar_texto)
    df_crisis = df_crisis.dropna(subset=['text', 'class_label'])

    print("Traduciendo crisis_dataset al español...")
    df_crisis['text'] = traducir_texto(df_crisis['text'].tolist(), model, tokenizer, device)

    df_crisis = sobremuestreo_aleatorio(df_crisis, 'class_label')
    columnas_crisis = ['text', 'class_label']
    df_crisis = df_crisis[columnas_crisis]
    df_crisis.to_csv(os.path.join(DATA_DIR, "crisis_limpio.csv"), index=False)
    print("crisis_dataset limpio guardado como crisis_limpio.csv")

# Ejecutar
if __name__ == "__main__":
    descargar_datasets()
    procesar_datasets()