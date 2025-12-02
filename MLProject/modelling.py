import pandas as pd
import os
import sys
from pathlib import Path
import time

# Import tambahan untuk fallback split data
from sklearn.model_selection import train_test_split 

# Import pustaka Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score 

# Import MLflow
import mlflow
import mlflow.sklearn
# Import untuk logging dataset terstruktur (best practice)
from mlflow.data.pandas_dataset import PandasDataset


# Path data yang sudah dipreproses
# Menggunakan path relatif, karena dipanggil dari MLProject
PREPROCESSED_DATA_PATH = Path('.') / 'amazon_preprocessed.csv' 

# Set nama model yang akan didaftarkan di MLflow
REGISTERED_MODEL_NAME_BASIC = "Sentiment_LR_Basic"


# --- PERBAIKAN: Set encoding output konsol ke UTF-8 ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
# ------------------------------------------------------


def load_data(path: Path):
    """Memuat data training dan testing dari file CSV."""
    print(f"Mencoba memuat data dari: {path}")
    df = pd.read_csv(path)
    
    # Periksa apakah ada kolom 'split' untuk pembagian data yang sudah ada
    if 'split' in df.columns:
        print("Kolom 'split' ditemukan. Memuat data berdasarkan kolom tersebut.")
        df_train = df[df['split'] == 'train'].copy()
        df_test = df[df['split'] == 'test'].copy()
    else:
        # Fallback: jika kolom 'split' tidak ada, gunakan train_test_split
        print("Kolom 'split' tidak ditemukan. Melakukan train_test_split 80/20.")
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])

    print(f"Data Loaded: Train Samples={len(df_train)}, Test Samples={len(df_test)}")
    return df_train, df_test


def run_training(data_path):
    """Fungsi utama untuk menjalankan pipeline training dan logging MLflow."""
    
    # --- PERBAIKAN UTAMA UNTUK CONFLICT RUN ID ---
    # Ketika dijalankan oleh 'mlflow run', kita TIDAK boleh memanggil mlflow.start_run()
    # atau mlflow.set_experiment(). Kita hanya perlu menggunakan run yang sudah aktif.
    # Logging akan otomatis terjadi pada run yang sudah dimulai oleh CLI 'mlflow run'.
    try:
        # Load Data
        df_train, df_test = load_data(Path(data_path))

        X_train, y_train = df_train['review'], df_train['sentiment']
        X_test, y_test = df_test['review'], df_test['sentiment']

        # Log Dataset terstruktur (Best Practice MLflow)
        # Mendapatkan active run
        active_run = mlflow.active_run()
        if active_run:
            print(f"MLflow Run ID Aktif: {active_run.info.run_id}")
            # Logging dataset sebagai PandasDataset
            train_dataset = PandasDataset(df=df_train, source=data_path, name="amazon_train_data")
            mlflow.log_input(train_dataset, context="training")
            test_dataset = PandasDataset(df=df_test, source=data_path, name="amazon_test_data")
            mlflow.log_input(test_dataset, context="validation")
        
        print("------------------------------------------------")
        print("MLflow Autologging AKTIF.")
        print("------------------------------------------------")

        # Aktifkan Autologging untuk Scikit-learn
        mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False, registered_model_name=REGISTERED_MODEL_NAME_BASIC)

        # 1. Pipeline: TF-IDF + Logistic Regression
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # 2. Training
        model.fit(X_train, y_train)

        # 3. Prediction & Evaluation
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log manual metrik untuk memastikan
        mlflow.log_metric("test_accuracy_manual", accuracy)
        mlflow.log_metric("test_f1_score_manual", f1)

        print(f"✅ Training Selesai.")
        print(f"Akurasi Test: {accuracy:.4f}")
        print(f"F1 Score Test: {f1:.4f}")

        # Log model (Model ini sudah dilog oleh Autologging, tapi kita tambahkan log eksplisit jika Autologging dimatikan)
        # mlflow.sklearn.log_model(model, "model")
            
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Pastikan Anda sudah menjalankan preprocessing dan file data ada di lokasi yang benar.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
            
if __name__ == "__main__":
    # Mengambil argumen data_path dari CLI, disediakan oleh MLproject
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=PREPROCESSED_DATA_PATH, help="Path to the preprocessed data file.")
    args = parser.parse_args()
    
    # --- PERBAIKAN: Hapus set_tracking_uri jika sudah diatur di CLI (best practice di MLProject) ---
    # Log the tracking URI to confirm the setting from YAML/CLI is active
    print(f"MLflow Tracking URI diatur ke folder lokal: {mlflow.get_tracking_uri()}")

    run_training(args.data_path)
