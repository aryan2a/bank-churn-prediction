FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements-app.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt -c constraints.txt

COPY churn ./churn
COPY app ./app
COPY .streamlit ./.streamlit

RUN useradd --create-home appuser
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]