import os
import chromadb

# 1. Connexion
client = chromadb.PersistentClient(path="./chroma_db")

print("\n=========================================")
print("📍 DIAGNOSTIC DE DOSSIER WINDOWS")
print("=========================================")
# Affiche le chemin complet sur ton disque dur
print(f"📁 Chemin absolu utilisé : {os.path.abspath('./chroma_db')}")

# Affiche toutes les collections présentes dans ce dossier spécifique
collections = client.list_collections()
print(f"🗂️ Collections présentes ici : {[col.name for col in collections]}")
print("=========================================\n")