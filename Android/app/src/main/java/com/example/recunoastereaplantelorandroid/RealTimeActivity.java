/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.recunoastereaplantelorandroid;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;


import java.io.IOException;

import com.example.recunoastereaplantelorandroid.ml.ResnetPre6;
import com.example.recunoastereaplantelorandroid.utils.Logger;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class  RealTimeActivity extends CameraActivity{
    private static final Logger LOGGER = new Logger();
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    private Bitmap rgbFrameBitmap = null;
    private long lastProcessingTimeMs;
    private ImageProcessor imageProcessor;


    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {

        recreateClassifier();
        if (reteaNeuronala == null) {
            LOGGER.e("Eroare generare clasificator");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    }

    @Override
    protected void processImage() {
        if (reteaNeuronala == null)
            return;
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        Log.i("Test Vlad", "Aici");
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();
                        TensorImage tImage = new TensorImage(DataType.FLOAT32);

                        // Analysis code for every frame
                        // Preprocess the image
                        tImage.load(rgbFrameBitmap);
                        tImage = imageProcessor.process(tImage);

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, reteaNeuronala.getImgSize(), reteaNeuronala.getImgSize(), 3}, DataType.FLOAT32);
                        inputFeature0.loadBuffer(tImage.getBuffer());


                        float[] rezultate = reteaNeuronala.processImage(inputFeature0);

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showResultsInBottomSheet(rezultate);
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                        readyForNextImage();
                    }
                });
    }


    private void recreateClassifier()
    {
        imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(reteaNeuronala.getImgSize(), reteaNeuronala.getImgSize(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0, 255))
                        .build();


    }
}
