# OnDevice CameraImage Classification
This is the base project where a basic Android app running Machine Learning model (Tensorflow Lite) on Camera feed. 
This prject can be used to run any machine learning model that takes camera feed as Input.
To do so, just replace the ```tflite``` model in the asset folder.

##Usage
1. Add the following permissions in the ```Manifest.xml``` file.
    ```<uses-permission android:name="android.permission.CAMERA"/>
       <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
    ```
2. Add the following dependencies in the ```gradle``` file. 
3. 
