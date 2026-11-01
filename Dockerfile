FROM python:3.9
# Directorio de trabajo
WORKDIR /app
# Copiar archivos
COPY . .
# Instalar dependencias
RUN pip install fastapi uvicorn scikit-learn pandas joblib
# Exponer puerto
EXPOSE 80
# Comando de ejecución
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]