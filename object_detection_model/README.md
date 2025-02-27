
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

1. Make sure you run the following command from the path **/numaware/object_detection_model/**
2. In the **name** argument pass the following string **initials-exp**. For example, if Marco Garcia is training than place **name=mg-exp**
3. In the **project** argument pass the following string **.**. For example, **project=.**
4. For the other arguments, read the ultralytics training api 


```
yolo detect train data=$(pwd)/data/processed/data.yaml model=yolo11n.yaml epochs=2 imgsz=640 project=./local/ name=mg-exp save=True
```

## Setting Up A Containerized Enviroment

This is optional. If you want a Containerized enviroment than follow the following steps. Also make sure to install docker. 

1. Build the image
```
docker build -t detect:1.0 .
```
2. Create the container from the image

This is just to shorten the command
```
run="docker run -it --mount type=bind,src=$(pwd),dst=/object_detection_model/ detect:1.0"
```
```
$run /bin/bash
```
3. Finally, from inside your container start training
```
yolo detect train data=$(pwd)/data/processed/data.yaml model=yolo11n.yaml epochs=2 imgsz=640 project=./local/ name=mg-exp save=True
```
