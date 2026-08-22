import chromadb

# 1. Connexion au dossier local de ta base de données
client = chromadb.PersistentClient(path="chroma_db/chroma_db")

# 2. Récupérer ta collection active
# (Assure-toi d'utiliser le nom exact de ta collection, par exemple "arabic_documents")
collection = client.get_collection(name="arabic_documents")

# 3. Récupérer et afficher tous les documents
data = collection.get()

print(f"Nombre total de chunks trouvés : {len(data['ids'])}")
print("-" * 40)

for i in range(len(data['documents'])):
    print(f"ID : {data['ids'][i]}")
    print(f"Texte du chunk : {data['documents'][i]}")
    if data['metadatas'] and data['metadatas'][i]:
        print(f"Métadonnées : {data['metadatas'][i]}")
    print("-" * 40)