package com.example.imageclassificationcamera.tflite;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.NonNull;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class Classifier {

    private Interpreter interpreter;
    private Interpreter.Options modelOptions;
    private MappedByteBuffer tfLiteModel;
    private List<String> associated_labels;
    private ImageProcessor imageProcessor;
    private TensorImage inputImageBuffer;
    private TensorLabel tensorLabel;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;

    private int INPUT_SIZE = 3;
    private int PIXEL_SIZE = 3;
    private int IMAGE_MEAN = 3;
    private float IMAGE_STD = 255.0f;
    private int MAX_RESULTS = 3;
    private float THRESHOLD = 0.4f;
    private int IMAGESIZE_X;
    private int IMAGESIZE_Y;


    public Classifier(Context context, String modelPath, String labelPath) {
        try {
            // Loading model file from the model path
            tfLiteModel = FileUtil.loadMappedFile(context, modelPath);

            // Setting the options
            modelOptions = new Interpreter.Options();
            //modelOptions.addDelegate(new NnApiDelegate());
            //modelOptions.addDelegate(new GpuDelegate());
            modelOptions.setUseNNAPI(true);
            modelOptions.setNumThreads(5);

            // Initializing the tflite model
            interpreter = new Interpreter(tfLiteModel, modelOptions);

            // Setting the input image Tensor
            int imageTensorIndex = 0;
            int[] imageShape = interpreter.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
            IMAGESIZE_X = imageShape[1];
            IMAGESIZE_Y = imageShape[2];
            DataType imageDataType = interpreter.getInputTensor(imageTensorIndex).dataType();

            int probabilityTensorIndex = 0;
            int[] probabilityShape =
                    interpreter.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
            DataType probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType();

            // Creates the input tensor.
            inputImageBuffer = new TensorImage(imageDataType);

            // Creates the output tensor
            outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);


            // Loading the lable file
            associated_labels = FileUtil.loadLabels(context, labelPath);

            // Setting the image processor
            imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(IMAGESIZE_Y, IMAGESIZE_X, ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                            .build();

        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model", e);
        }
    }

    public String recognize(Bitmap bitmap) {
        long startTime = SystemClock.uptimeMillis();
        inputImageBuffer.load(bitmap);
        inputImageBuffer = imageProcessor.process(inputImageBuffer);

        interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer());

        tensorLabel = new TensorLabel(associated_labels,
                outputProbabilityBuffer);

        Map<String, Float> floatMap = tensorLabel.getMapWithFloatValue();
        long endTime = SystemClock.uptimeMillis();

        String result = "Prediction is " + getTopKProbability(floatMap) + "\nInference Time " + (endTime-startTime)+" ms";
        return result;

    }

    private String getTopKProbability(Map<String, Float> floatMap) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : floatMap.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue()));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions.get(0).getTitle();
    }


    public class Recognition {
        public String title = "";
        public String id = "";
        public float confidence = 0.0f;

        public Recognition(String title, String id, float confidence) {
            this.title = title;
            this.id = id;
            this.confidence = confidence;
        }

        @NonNull
        @Override
        public String toString() {
            return "Title: "+title+" Confidence: "+confidence;
        }

        public String getTitle() {
            return title;
        }

        public void setTitle(String title) {
            this.title = title;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public float getConfidence() {
            return confidence;
        }

        public void setConfidence(float confidence) {
            this.confidence = confidence;
        }
    }
}
