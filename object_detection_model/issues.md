## Common Issues

The following list contains common issues and solutions. If you run into an issue and fix it please log it here.

- 2/26/25 mg 

    **Issue**: After setting up the project and trying to run the ultralytics api the following error occurs. 

    *ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory*

    **Solution**: Install the Portable API from glib wrapping system threads.

    ```
    sudo zypper in libgthread-2_0-0
    ```

- 3/13/25 jm 

    **Issue**: When trying to run collect.py for joint data collection you will run into the following error:

    *ModuleNotFoundError: No module named 'tkinter'*

    **Solution**: Zypper install tk:

    ```
    sudo zypper install python311-tk
    ```
