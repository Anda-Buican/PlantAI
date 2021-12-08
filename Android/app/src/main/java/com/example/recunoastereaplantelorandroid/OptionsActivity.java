package com.example.recunoastereaplantelorandroid;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.Spinner;

import java.io.IOException;

public class OptionsActivity extends AppCompatActivity {

    private Spinner mModelAles = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_options);

        handleButoane();

        mModelAles = (Spinner) findViewById(R.id.model_spinner);

        // Afisare setari alese de user
        updateUi();
    }

    private void updateUi()
    {
        AppSavedData appSavedData = AppSavedData.load(this);

        mModelAles.setSelection(appSavedData.reteaAleasa.ordinal());
    }

    // Functie care se ocupa de onClick butoane
    private void handleButoane()
    {
        // Inapoi
        Button bBack = findViewById(R.id.button_back);
        bBack.setOnClickListener(view -> {
            finish();
        });

        // Sterge toate pozele TODO
        Button bDelete = findViewById(R.id.button_delete_pics);
        bDelete.setOnClickListener(view -> {
            // Alert sigur se vrea asta ?
            new AlertDialog.Builder(this)
                    .setTitle("Atentie")
                    .setMessage("Sunteti sigur ca doriti sa stergeti pozele adaugate in galerie?")
                    .setPositiveButton(android.R.string.yes, (dialog, which) -> {
                        // Continue with delete operation
                        AppSavedData appSavedData = AppSavedData.load(this);
                        appSavedData.poze.clear();
                        try {
                            appSavedData.save(this);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    })
                    .setNegativeButton(android.R.string.no, null)
                    .setIcon(android.R.drawable.ic_dialog_alert)
                    .show();
        });

        // Salveaza
        Button bSave = findViewById(R.id.button_save);
        bSave.setOnClickListener(view -> {
            AppSavedData appSavedData = AppSavedData.load(this);
            appSavedData.reteaAleasa = ReteaNeuronala.TipRetea.values()[mModelAles.getSelectedItemPosition()];
            try {
                appSavedData.save(this);
                // Alert sigur se vrea asta ?
                new AlertDialog.Builder(this)
                        .setTitle(R.string.success)
                        .setMessage("Modificarile au fost salvate")
                        .setPositiveButton(android.R.string.yes, (dialog, which) -> {

                        })
//                        .setIcon(android.R.drawable.ic_dialog_alert) TODO
                        .show();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }
}