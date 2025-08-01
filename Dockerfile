# Dockerfile
FROM python:3.13-slim

# 1. Establece el directorio de trabajo
WORKDIR /app

# 2. Copia dependencias y las instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copia el código y los artefactos
COPY services/ services/
COPY data/vectorized/ data/vectorized/

# 4. Expone el puerto que usará Uvicorn
EXPOSE 8000

# 5. Comando de arranque
CMD ["uvicorn", "services.recs_api:app", "--host", "0.0.0.0", "--port", "8000"]
