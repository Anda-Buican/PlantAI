package com.example.recunoastereaplantelorandroid;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.GridView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.example.recunoastereaplantelorandroid.utils.Constants;

import java.util.Vector;

public class PhotoGalleryActivity extends AppCompatActivity {

    private GridView gridView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo_gallery);

        handleButoane();

        gridView = (GridView) findViewById(R.id.grid);

        // Luam datele despre clasa dorita
        int category = getIntent().getIntExtra("CLASS_ID",0);
        PhotoGridAdapter adapter = new PhotoGridAdapter(this, handleGalleryTiles(category));
        gridView.setAdapter(adapter);
    }

    private Vector<ImageData> handleGalleryTiles(int category)
    {
        Vector<ImageData> imageDataVector = new Vector<>();

        // Datele despre poze
        AppSavedData appSavedData = AppSavedData.load(this);

        // Mergem peste fiecare poza si verificam in ce clasa face parte
        for(ImageData imageData : appSavedData.poze)
        {
            if(imageData.category == category)
            {
                imageDataVector.add(imageData);
            }
        }

        return imageDataVector;
    }


    // Functie care se ocupa de onClick butoane
    private void handleButoane()
    {
        // Inapoi
        Button bBack = findViewById(R.id.button_back);
        bBack.setOnClickListener(view -> {
            finish();
        });

    }
}