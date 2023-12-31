# summarize-lang-chain

## Setup

### Creating virtual environment.

- Run ```pip install virtualenv```.

- Run ```python -m venv env```.

- Run ```env/Scripts/activate.bat``` file when using cmd or ```env/Scripts/Activate.ps1``` when using powershell or ```source env/bin/activate``` in linux.

- Run ```pip install -r requirements.txt```

- *Note:- In order to activate environment in power shell run ```set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser``` to activate script running priveleges.

### Create .env file
- Add ```OPENAI_API_KEY``` with your actual openai api key.

### Running app

- Run ```streamlit run .\main.py``` to run the app in browser.

- Run ```python .\summarize.py``` to run the app in console.

### Running docker

- Install docker from https://docs.docker.com/desktop/install/windows-install/.

- Run ```docker build -t summarize-document-lang-chain .``` to build docker file.

- Run ```docker run -d -p 8501:8501 summarize-document-lang-chain:latest``` to run the docker image in a container locally.