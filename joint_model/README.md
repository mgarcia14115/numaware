
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


## Training Our Joint Model
1. Make sure you are in the **numaware/joint_model/src/uafscs/** directory
2. The below command is only the **requred** command line arguments. Type **python3.11 main.py -h** to view the optional commands.

```
python3.11 main.py --modelname=baseline --train_labels=../../data/processed/train_data.csv --val_labels=../../data/processed/test_data.csv --target=cartesians --midpoint=yolo
```
