## End to End Stroke Prediction

* Create virtual environment:
	  `conda create -p venv python==3.8 -y`

* Activate virtual environment:
	  `conda activate venv/`
* Install requirements:
	`pip install -r requirements.txt`

* Git Commands
	```
		git commit -m "<Message>"
		git push -u origin main
	```
## Problem Statement: [Kaggle DataSet](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Notebook Solution: [ABDULRAHMAN KHALID](https://www.kaggle.com/code/abdulrahmankhaled1/stroke-prediction-ensemble-learning-for-beginners/notebook)

### Sample Records for Web
* Positive Record: Female,79.0,0,1,    No,Private,Urban,    205.33,31.0,smokes,1
* Negative Record: Female,32.0,0,0,   Yes,Private,Rural,   76.13,29.9,smokes,0

## Launch MLFlow ui
```
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## To train ML Model
```
phthon src/pipeline/train_pipeline.py
```

## To get prediction
Launch Flask server and then open browser `http://localhost:8080/predictdata`, Fill form and then click on `Predict Stroke` button.
