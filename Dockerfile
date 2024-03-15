# Use the official Python base image

FROM python:3.9

ENV HOST 0.0.0.0

EXPOSE 8080

# Set the working directory inside the container

WORKDIR /app

COPY . ./


RUN apt-get update && apt-get install -y libmupdf-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set the command to run the app

CMD ["streamlit", "run", "app/LaudeChat.py", "--server.port=8080", "--server.address=0.0.0.0"]