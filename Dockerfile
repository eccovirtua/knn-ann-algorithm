# Dockerfile
FROM python:3.10-slim

# 1. Configurar directorio de trabajo
WORKDIR /app

# 2. Copiar requerimientos e instalar (aprovecha caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar el resto del código
COPY . .

# 4. Exponer el puerto que usa Cloud Run (por defecto 8080)
ENV PORT=8080

# 5. Comando de arranque (Uvicorn)
# Asegúrate de que 'recs_api' coincida con el nombre de tu archivo .py
CMD ["uvicorn", "recs_api:app", "--host", "0.0.0.0", "--port", "8080"]