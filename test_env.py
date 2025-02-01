from dotenv import load_dotenv
import os

# Carga el archivo .env (por defecto busca en el directorio actual)
load_dotenv()

# Imprime los valores para verificar
print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
print("SLACK_BOT_TOKEN:", os.environ.get("SLACK_BOT_TOKEN"))
print("SLACK_SIGNING_SECRET:", os.environ.get("SLACK_SIGNING_SECRET"))
print("PORT:", os.environ.get("PORT"))
