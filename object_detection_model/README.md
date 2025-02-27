
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


## Common Issues

The following list contains common issues and solutions. If you run into an issue and fix it please log it here.

- 2/26/25 mg 

    **Issue**: After setting up the project and trying to run the ultralytics api the following error occurs. 

    *ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory*

    **Solution**: Install the Portable API from glib wrapping system threads.

    ```
    sudo zypper in libgthread-2_0-0
    ```

