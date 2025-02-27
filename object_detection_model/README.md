
## Setup Project
The following steps show how to set up this project. The assumption is that you are running linux. 


1. Clone the project and install requirements
```
git clone git@github.com:mgarcia14115/numaware.git 
```

2. Install python and create a virtual enviroment
```
sudo zypper in python311
```

```
python3.11 -m venv .venv
```

```
source .venv/bin/activate
```

3. Install requirements

```
pip install --upgrade pip 
```

```
pip install -r requirements.txt 
```


## Training A Yolo Model
Go ./references/link.md and look for the ultralytics training api link. You can find the arguments to tune hyperameters.

Make you run this from the path /numaware/object_detection_model/

```
yolo detect train data=$(pwd)/data/processed/data.yaml model=yolo11n.yaml epochs=2 imgsz=640 project=. name=mg-exp save=true
```