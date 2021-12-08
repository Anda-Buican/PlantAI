package com.example.recunoastereaplantelorandroid;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.GridView;
import android.widget.Toast;

import com.example.recunoastereaplantelorandroid.utils.Constants;

import java.util.List;
import java.util.Vector;

public class ClassGalleryActivity extends AppCompatActivity {

    private GridView gridView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_class_gallery);

        handleButoane();

        gridView = (GridView) findViewById(R.id.grid);

        CategoryGridAdapter adapter = new CategoryGridAdapter(this, handleGalleryTiles());
        gridView.setAdapter(adapter);

        // Handle click pe grid
        gridView.setOnItemClickListener(new AdapterView.OnItemClickListener() {

            @Override
            public void onItemClick(AdapterView<?> parent, View v, int poz, long id) {
//                // Cerem dreptul de deschidere daca nu il avem deja
//                // Verificari de permisiune
//                if(checkSelfPermission(Manifest.permission.)==PackageManager.PERMISSION_DENIED){
//                    // daca nu ai permisiunea fa request
//                    String[] permission={Manifest.permission.READ_EXTERNAL_STORAGE};
//
//                    requestPermissions(permission,poz);
//                }
//                else{
                    // Luam obiectul apasat
                    PlantCategory p = (PlantCategory) gridView.getItemAtPosition(poz);

                    // Deschidem galeria pt clasa aleasa
                    Intent intent = new Intent(ClassGalleryActivity.this, PhotoGalleryActivity.class);
                    intent.setAction("android.permission.ACTION_OPEN_DOCUMENT");
                    intent.putExtra("CLASS_ID", p.getCategoryId());
                    startActivity(intent);
//                }


            }
        });
    }

    private Vector<PlantCategory> handleGalleryTiles()
    {
        Vector<PlantCategory> plantCategoryList = new Vector<>();

        // Datele despre poze
        AppSavedData appSavedData = AppSavedData.load(this);

        // Vector de frecventa
        int frecv[] = new int[Constants.clase.length];

        // Mergem peste fiecare poza si verificam in ce clasa face parte
        for(ImageData imageData : appSavedData.poze)
        {
            frecv[imageData.category]++;
        }

        // TODO de luat categoriile din un setting+datele lor
        int id =0;
        for(String denumire : Constants.clase)
        {
            plantCategoryList.add(new PlantCategory(id,denumire, frecv[id++]));
        }

        return plantCategoryList;
    }


    // Functie care se ocupa de onClick butoane
    private void handleButoane()
    {
        // Inapoi
        Button bBack = findViewById(R.id.button_back_gallery);
        bBack.setOnClickListener(view -> {
            finish();
        });

        // La clasificare
        Button bRealTime = findViewById(R.id.button_clasif_gallery);
        bRealTime.setOnClickListener(view -> {
            Intent intent = new Intent(ClassGalleryActivity.this, ClassifyActivity.class);
            startActivity(intent);
        });
    }

    // Handle pentru raspunsul la permisiuni
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        //metoda e creata cand utilizatorul apasa allow sau deny la permisiuni
        if(grantResults.length>0 && grantResults[0]== PackageManager.PERMISSION_GRANTED){
            // Luam obiectul apasat
            PlantCategory p = (PlantCategory) gridView.getItemAtPosition(requestCode);

            // Deschidem galeria pt clasa aleasa
            Intent intent = new Intent(ClassGalleryActivity.this, PhotoGalleryActivity.class);
            intent.putExtra("CLASS_ID", p.getCategoryId());
            startActivity(intent);
        }
        else{
            Toast.makeText(this, "Permission denied",Toast.LENGTH_SHORT).show();
        }
    }
}