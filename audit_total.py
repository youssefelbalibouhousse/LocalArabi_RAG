import os
import sqlite3
import chromadb

print("\n=========================================")
print("🔬 RAPPORT D'AUDIT CHROMA & SYSTEME")
print("=========================================")

# 1. Version de ChromaDB
print(f"📦 Version de ChromaDB utilisée : {chromadb.__version__}")

# 2. Vérification des fichiers sur le disque
db_path = "./chroma_db"
print(f"📁 Chemin du dossier : {os.path.abspath(db_path)}")
if os.path.exists(db_path):
    files = os.listdir(db_path)
    print(f"📄 Fichiers physiques trouvés dans ce dossier : {files}")
else:
    print("❌ Le dossier 'chroma_db' n'existe pas physiquement à cet endroit.")

# 3. Tentative de lecture directe SQL (Vérification physique)
sqlite_file = os.path.join(db_path, "chroma.sqlite3")
if os.path.exists(sqlite_file):
    print("\n🔍 Tentative de lecture physique de la base SQLite...")
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()

        # Récupérer la liste des tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📊 Tables SQL trouvées : {tables}")

        if "collections" in tables:
            cursor.execute("SELECT name, id FROM collections;")
            rows = cursor.fetchall()
            print(f"💾 Collections enregistrées en SQL : {rows}")
        else:
            print("⚠️ La table 'collections' n'existe pas dans ce fichier SQLite.")

        conn.close()
    except Exception as e:
        print(f"❌ Erreur lors de la lecture SQLite directe : {e}")
else:
    print("\n⚠️ Aucun fichier 'chroma.sqlite3' trouvé (ancien moteur de stockage ?)")

print("=========================================\n")