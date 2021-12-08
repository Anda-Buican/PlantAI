/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.recunoastereaplantelorandroid;

import android.app.Activity;
import android.content.Context;

import androidx.core.content.ContextCompat;

import com.example.recunoastereaplantelorandroid.ml.Inceptionv36;
import com.example.recunoastereaplantelorandroid.ml.Resnet34AugmC6;
import com.example.recunoastereaplantelorandroid.ml.ResnetPre6;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;

/** This TensorFlow Lite classifier works with the quantized MobileNet model. */
public class ReteaNeuronala {

  public enum TipRetea {
    RESNET34,
    RESNET50_PRE,
    INCEPTIONV2
  }

  private final TipRetea tipRetea;

  private ResnetPre6 resnetPre6 = null;
  private Inceptionv36 inceptionv36 = null;
  private Resnet34AugmC6 resnet34 = null;

  public ReteaNeuronala(TipRetea tipRetea, Context context) throws IOException {
    this.tipRetea = tipRetea;

    switch (tipRetea)
    {
      case RESNET50_PRE:
        resnetPre6 = ResnetPre6.newInstance(context);
      case RESNET34:
        resnet34 = Resnet34AugmC6.newInstance(context);
      case INCEPTIONV2:
        inceptionv36 = Inceptionv36.newInstance(context);
    }
  }

  public int getImgSize()
  {
    switch (tipRetea)
    {
      case RESNET34:
      case RESNET50_PRE:
        return 224;
      case INCEPTIONV2:
        return 299;
    }
    return -1;
  }

  public float[] processImage(TensorBuffer inputFeature0)
  {
    float[] rezultate = null;

    switch (tipRetea) {
      case RESNET50_PRE:
        rezultate = resnetPre6.process(inputFeature0).getOutputFeature0AsTensorBuffer().getFloatArray();
        break;
      case RESNET34:
        rezultate = resnet34.process(inputFeature0).getOutputFeature0AsTensorBuffer().getFloatArray();
        break;
      case INCEPTIONV2:
        rezultate = inceptionv36.process(inputFeature0).getOutputFeature0AsTensorBuffer().getFloatArray();
        break;
    }
    return rezultate;
  }

}
