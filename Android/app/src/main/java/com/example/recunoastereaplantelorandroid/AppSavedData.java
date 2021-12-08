package com.example.recunoastereaplantelorandroid;

import android.content.Context;
import android.net.Uri;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Date;
import java.util.Vector;

public class AppSavedData implements Serializable {
    public Vector<ImageData> poze;
    public ReteaNeuronala.TipRetea reteaAleasa;

    public AppSavedData()
    {
        poze = new Vector<>();
        reteaAleasa = ReteaNeuronala.TipRetea.RESNET34;
    }

    public static AppSavedData load(Context context)
    {
        try {
            FileInputStream fis = context.openFileInput("AppSavedData");
            ObjectInputStream is = new ObjectInputStream(fis);
            AppSavedData appData = (AppSavedData) is.readObject();
            is.close();
            fis.close();
            return appData;
        } catch (IOException | ClassNotFoundException e) {
            return new AppSavedData();
        }
    }

    public void save(Context context) throws IOException {
        FileOutputStream fos = context.openFileOutput("AppSavedData", Context.MODE_PRIVATE);
        ObjectOutputStream os = new ObjectOutputStream(fos);
        os.writeObject(this);
        os.close();
        fos.close();
    }
}

class ImageData implements Serializable{
    public String path;
    public Date dataAdaugare;
    public int category;

    public ImageData(String path,Date dataAdaugare, int category)
    {
        this.path = path;
        this.dataAdaugare = dataAdaugare;
        this.category = category;
    }
}
