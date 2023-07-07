FROM python:3.10.5

WORKDIR /app

COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy all the source code from current directory to image
COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]