FROM python:3.10.5

WORKDIR /app

COPY requirements.txt requirements.txt

# Install dependencies
# Added --no-cache-dir to resolve low ram capacity issue.
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the source code from current directory to image
COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]