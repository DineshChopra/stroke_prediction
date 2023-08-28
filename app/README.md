## Real time prediction

## Batch Prediction

## Launch web-app
* uvicorn app.main:app

## Build the container
* docker build -t stroke-prediction:real-time .
## Run the container:
* docker run --rm -p 80:80 stroke-prediction:real-time