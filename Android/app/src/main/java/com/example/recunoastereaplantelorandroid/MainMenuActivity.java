package com.example.recunoastereaplantelorandroid;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.Gallery;

public class MainMenuActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_menu);

        handleButoane();
    }

    // Functie care se ocupa de onClick butoane
    private void handleButoane()
    {
        // Clasificare
        Button bClasif = findViewById(R.id.button_add);
        bClasif.setOnClickListener(view -> {
            Intent intent = new Intent(MainMenuActivity.this, ClassifyActivity.class);
            startActivity(intent);
        });

        // Real Time
        Button bRealTime = findViewById(R.id.button_realTime);
        bRealTime.setOnClickListener(view -> {
            Intent intent = new Intent(MainMenuActivity.this, RealTimeActivity.class);
            startActivity(intent);
        });

        // Galerie
        Button bGallery = findViewById(R.id.button_gallery);
        bGallery.setOnClickListener(view -> {
            Intent intent = new Intent(MainMenuActivity.this, ClassGalleryActivity.class);
            startActivity(intent);
        });

        // Optiuni
        Button bOptiuni = findViewById(R.id.button_menu);
        bOptiuni.setOnClickListener(view -> {
            Intent intent = new Intent(MainMenuActivity.this, OptionsActivity.class);
            startActivity(intent);
        });

        // Exit
        Button bExit = findViewById(R.id.button_exit);
        bExit.setOnClickListener(view -> System.exit(0));
    }
}