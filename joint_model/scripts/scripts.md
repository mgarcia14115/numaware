## This is a list of all the scripts and how to use them. 

The prequistes to running these scripts are the following:

1. make a virtual env inside of the **numaware/joint_model/** directory. 

2. Activate it

3. Install the **numaware/joint_model/requirements.txt** file. Make sure its the one inside of the joint_model directory. 

```
pip install -r requirements.txt
```



---------------------------------------------------------------


- findCamera.py 
  
  **Purpose**: Find the index of a camera. 

  ```
  python3.11 findCamera.py
  ```

- collect.py
  **Purpose**: Collect Data for the joint model.

  1. Create an a directory that will hold an image directory and a data.csv file. 

  2. Save the path for both image directory and data.csv file

  I create enviroment variables to shorten the command. 
  ```
  img_dir=/absolute/path/to/images
  ```
  ```
  data_file=/absolute/path/to/data.csv
  ```
  Finally the actual command
  ```
  python3.11 collect.py $img_dir $data_file camera_index
  ```