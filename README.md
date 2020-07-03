# OnDevice CameraImage Classification
This is the base project where a basic Android app running Machine Learning model (Tensorflow Lite) on Camera feed. 
This prject can be used to run any machine learning model that takes camera feed as Input.
To do so, just replace the ```tflite``` model in the asset folder.

## Usage
1. Add the following permissions in the `Manifest.xml` file.
```<uses-permission android:name="android.permission.CAMERA"/> 
   <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/> 
   ```
       
2. Add the following dependencies in the `gradle` file. It uses CameraX libbrary for accessing, previewing and analying. 
```
 implementation "androidx.camera:camera-core:1.0.0-alpha02"
 implementation "androidx.camera:camera-camera2:1.0.0-alpha02"

 implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
 implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly'
 implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly'

```
3. Build and Run the app. 
