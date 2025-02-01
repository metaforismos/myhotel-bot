import os
import glob
import numpy as np
from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from apscheduler.schedulers.background import BackgroundScheduler
import openai
from dotenv import load_dotenv
import sys

# ---------------------------------------------------
# Cargar las variables de entorno desde el archivo .env explícitamente
# ---------------------------------------------------
load_dotenv(dotenv_path=".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
PORT = int(os.environ.get("PORT", 3000))

if not all([OPENAI_API_KEY, SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET]):
    print("Error: Por favor, configura las variables de entorno: OPENAI_API_KEY, SLACK_BOT_TOKEN y SLACK_SIGNING_SECRET", flush=True)
    sys.exit(1)

# Configurar la clave de la API de OpenAI
openai.api_key = OPENAI_API_KEY

# ---------------------------------------------------
# Funciones para manejo de embeddings y búsqueda semántica
# ---------------------------------------------------

knowledge_base = []  # Lista que contendrá la base de conocimiento con sus embeddings

def get_embedding(text):
    """
    Calcula el embedding de un texto usando la API de OpenAI.
    Se envía el texto como una lista (requerido por la interfaz) y se convierte la respuesta a un diccionario.
    """
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        result = response.to_dict()
        embedding = result['data'][0]['embedding']
        return np.array(embedding)
    except Exception as e:
        print(f"Error obteniendo embedding: {e}", flush=True)
        return None

def load_knowledge_base():
    """
    Lee todos los archivos de la carpeta 'knowledge_files' y calcula sus embeddings.
    Se espera que los archivos sean de texto (.txt).
    """
    global knowledge_base
    knowledge_base = []
    file_paths = glob.glob("knowledge_files/*.txt")
    if not file_paths:
        print("No se encontraron archivos en 'knowledge_files'. Asegúrate de que existan archivos .txt en esa carpeta.", flush=True)
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            embedding = get_embedding(content)
            if embedding is not None:
                knowledge_base.append({
                    "filename": os.path.basename(file_path),
                    "text": content,
                    "embedding": embedding
                })
                print(f"Cargado y procesado: {file_path}", flush=True)
        except Exception as e:
            print(f"Error procesando {file_path}: {e}", flush=True)

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_knowledge(query, top_n=1):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return ""
    similarities = []
    for entry in knowledge_base:
        sim = cosine_similarity(query_embedding, entry["embedding"])
        similarities.append(sim)
    if similarities:
        best_index = int(np.argmax(similarities))
        best_entry = knowledge_base[best_index]
        print(f"Mejor coincidencia: {best_entry['filename']} con similitud {similarities[best_index]:.3f}", flush=True)
        return best_entry["text"]
    return ""

# ---------------------------------------------------
# Funciones para generar respuesta (con y sin streaming)
# ---------------------------------------------------

def generate_response_stream(user_query, context):
    """
    Genera una respuesta en modo streaming utilizando OpenAI, mostrando el contenido progresivamente.
    Se le indica al modelo que responda de forma concisa, precisa e informativa, limitándose a lo estrictamente solicitado.
    """
    prompt = f"""
Utiliza el siguiente contexto como base, pero responde únicamente a la pregunta realizada, sin incluir información adicional no solicitada.
Si la pregunta es "¿Qué es myHotel?", explica brevemente qué es y cuál es el beneficio principal para los hoteles al usar sus módulos o funcionalidades.
Si la pregunta es sobre otro tema (por ejemplo, la visión del 2025, la misión, etc.), responde únicamente en ese contexto.

Contexto:
{context}

Pregunta: {user_query}

Respuesta:
"""
    try:
        response_stream = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "Eres un asistente para el equipo de myHotel. Responde de forma concisa, precisa e informativa, "
                    "limitándote a la información estrictamente requerida según la pregunta. Explica siempre el beneficio que "
                    "tiene para los hoteles usar módulos o funcionalidades específicas del software myHotel. No incluyas información adicional."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            stream=True
        )
        full_response = ""
        print("Respuesta:", end=" ", flush=True)
        for chunk in response_stream:
            delta = chunk['choices'][0].get('delta', {})
            if 'content' in delta:
                texto = delta['content']
                full_response += texto
                print(texto, end="", flush=True)
        print()
        return full_response
    except Exception as e:
        print(f"\nError generando respuesta en modo streaming: {e}", flush=True)
        return "Lo siento, ocurrió un error al generar la respuesta."

def generate_response(user_query, context):
    """
    Genera una respuesta de forma no progresiva (sin streaming) utilizando OpenAI.
    """
    prompt = f"""
Utiliza el siguiente contexto como base, pero responde únicamente a la pregunta realizada, sin incluir información adicional no solicitada.
Si la pregunta es "¿Qué es myHotel?", explica brevemente qué es y cuál es el beneficio principal para los hoteles al usar sus módulos o funcionalidades.
Si la pregunta es sobre otro tema (por ejemplo, la visión del 2025, la misión, etc.), responde únicamente en ese contexto.

Contexto:
{context}

Pregunta: {user_query}

Respuesta:
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "Eres un asistente para el equipo de myHotel. Responde de forma concisa, precisa e informativa, "
                    "limitándote a la información estrictamente requerida según la pregunta. Explica siempre el beneficio que "
                    "tiene para los hoteles usar módulos o funcionalidades específicas del software myHotel. No incluyas información adicional."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            n=1
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except Exception as e:
        print(f"Error generando respuesta: {e}", flush=True)
        return "Lo siento, ocurrió un error al generar la respuesta."

# ---------------------------------------------------
# Configuración de Slack Bolt y Flask
# ---------------------------------------------------

slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
flask_app = Flask(__name__)
handler = SlackRequestHandler(slack_app)

@flask_app.route("/")
def index():
    return "Server is up and running!"

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

def tarea_programada():
    """
    Envía un mensaje programado a un canal de Slack recordando revisar las actualizaciones del producto.
    """
    canal = "#general"
    mensaje = "¡Recordatorio! No olviden revisar las actualizaciones y novedades de myHotel."
    try:
        slack_app.client.chat_postMessage(channel=canal, text=mensaje)
        print("Mensaje programado enviado.", flush=True)
    except Exception as e:
        print(f"Error enviando mensaje programado: {e}", flush=True)

scheduler = BackgroundScheduler()
scheduler.add_job(tarea_programada, 'cron', hour=9, minute=0)
scheduler.start()

# ---------------------------------------------------
# Bloque principal: modo interactivo o servidor
# ---------------------------------------------------
if __name__ == "__main__":
    print("Inicio del script main.py", flush=True)
    print("Cargando base de conocimiento...", flush=True)
    load_knowledge_base()
    print("Base de conocimiento cargada.", flush=True)
    
    # Modo interactivo: si se pasa el argumento "terminal", se usa streaming para mostrar la respuesta progresivamente.
    if len(sys.argv) > 1 and sys.argv[1] == "terminal":
        while True:
            query = input("Ingrese su pregunta (o 'salir' para terminar): ")
            if query.lower() in ["salir", "exit"]:
                break
            context = search_knowledge(query)
            # Usamos la función de streaming para generar la respuesta en tiempo real.
            generate_response_stream(query, context)
    else:
        print(f"Ejecutando servidor en el puerto {PORT}...", flush=True)
        flask_app.run(host="0.0.0.0", port=PORT)
