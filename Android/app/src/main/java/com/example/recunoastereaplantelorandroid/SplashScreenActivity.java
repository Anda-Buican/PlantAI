package com.example.recunoastereaplantelorandroid;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

import androidx.appcompat.app.AppCompatActivity;

public class SplashScreenActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Asteapta 2 secude si apoi mergi la meniu
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                // Porneste meniul
                startActivity(new Intent(SplashScreenActivity.this, MainMenuActivity.class));
                // Inchide pagina curenta
                finish();
            }
        }, 2000);
    }
}
