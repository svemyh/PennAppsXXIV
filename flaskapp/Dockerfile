#FROM python:3.9-alpine
FROM --platform=linux/x86_64 python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


#RUN apk update
#RUN apk add make automake gcc g++ subversion python3-dev

RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install tensorflow


#RUN pip install https://github.com/diyor28/tf-docker-m1/releases/download/v1.0.0/tensorflow-2.8.0-cp39-cp39-linux_aarch64.whl

RUN pip install -r requirements.txt
# Copy the Flask app source code into the container
COPY . .

# Expose the port your Flask app will run on
EXPOSE 5000

# Define the command to run your Flask app
CMD ["python", "app.py"]
