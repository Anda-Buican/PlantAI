package com.example.recunoastereaplantelorandroid;

import android.Manifest;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.DialogFragment;

import com.example.recunoastereaplantelorandroid.utils.ArrayChoiceDialog;
import com.example.recunoastereaplantelorandroid.utils.Constants;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Comparator;
import java.util.Vector;


public class ClassifyActivity extends AppCompatActivity implements ArrayChoiceDialog.ArrayChoiceDialogListener{

    private static final int PERMISSION_CODE = 1000; // doar userul are permisiune
    private static final int IMAGE_CAPTURE_CODE = 1001;
    private static final int IMAGE_GALLERY_CODE = 1002;

    Button mCaptureBtn;
    Button mClassifierBtn;
    ImageView mImageView;
    RelativeLayout mZonaActiuni;
    GridLayout mZonaClasificare;
    Boolean existaPoza = Boolean.FALSE;
    Interpreter tflite;
    EditText inputNumber;
    Vector<TextView> mlistaClase;
    Vector<TextView> mlistaProcent;
    Vector<ImageView> mlistaImagini;

    // Variabile pentru adaugare poza
    private int clasaGasita = -1;
    private Uri path = null;

    Uri image_uri;

    private ReteaNeuronala reteaNeuronala = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.image_view);
        mCaptureBtn = findViewById(R.id.capture_image_btn);
        mClassifierBtn = findViewById(R.id.classifier_image_btn);
        mZonaActiuni     = findViewById(R.id.zonaActiuni);
        mZonaClasificare     = findViewById(R.id.zonaClasificare);

        mlistaClase = new Vector<>();
        mlistaClase.add((TextView)findViewById(R.id.textClasificare1));
        mlistaClase.add((TextView)findViewById(R.id.textClasificare2));
        mlistaClase.add((TextView)findViewById(R.id.textClasificare3));

        mlistaProcent = new Vector<>();
        mlistaProcent.add((TextView)findViewById(R.id.textProcent1));
        mlistaProcent.add((TextView)findViewById(R.id.textProcent2));
        mlistaProcent.add((TextView)findViewById(R.id.textProcent3));

        mlistaImagini = new Vector<>();
        mlistaImagini.add((ImageView) findViewById(R.id.imagineClasificare1));
        mlistaImagini.add((ImageView)findViewById(R.id.imagineClasificare2));
        mlistaImagini.add((ImageView)findViewById(R.id.imagineClasificare3));

        handleButoane();

        // Deschidem datele salvate
        AppSavedData appSavedData = AppSavedData.load(this);

        // Setam datele specifice fiecarei retele
        try {
            reteaNeuronala = new ReteaNeuronala(appSavedData.reteaAleasa,this);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void deschideCamera(){
        ContentValues values= new ContentValues();
        values.put(MediaStore.Images.Media.TITLE,"New Picture");
        values.put(MediaStore.Images.Media.DESCRIPTION,"FROM CAMERA");
        //insereaza noua imagine in memoria externa
        image_uri=getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,values);
        //deschide actiunea camera
        Intent actiuneCamera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        actiuneCamera.putExtra(MediaStore.EXTRA_OUTPUT,image_uri);
        startActivityForResult(actiuneCamera,IMAGE_CAPTURE_CODE);
    }

    // Handle pentru raspunsul la permisiuni
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        //metoda e creata cand utilizatorul apasa allow sau deny la permisiuni
        switch(requestCode){
            case PERMISSION_CODE:{
                if(grantResults.length>0 && grantResults[0]== PackageManager.PERMISSION_GRANTED){
                    deschideCamera();
                }
                else{
                    Toast.makeText(this, "Permission denied",Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {

            // Raspuns poza din camera
            if(requestCode == IMAGE_CAPTURE_CODE)
            {
                //set image captured to our imageview
                mImageView.setImageURI(image_uri);
                path = image_uri;
                existaPoza=Boolean.TRUE;
            }
            // Raspuns poza din galerie
            else if(requestCode == IMAGE_GALLERY_CODE)
            {
                assert data != null;
                Uri selectedImageUri = data.getData();
                if (null != selectedImageUri) {
                    final int takeFlags = data.getFlags()
                            & (Intent.FLAG_GRANT_READ_URI_PERMISSION
                            | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
                    this.getContentResolver().takePersistableUriPermission(selectedImageUri, takeFlags);
                    mImageView.setImageURI(selectedImageUri);
                    path = selectedImageUri;
                    existaPoza=Boolean.TRUE;
                }
            }

            // Activam butonul clasificare
            mClassifierBtn.setEnabled(true);
        }
    }

    @Override
    public void onDialogClick(DialogFragment dialog, int option) {
        if(option == 0)
        {
            // Utilizatorul a ales camera
            // Verificari de permisiune
            if(checkSelfPermission(Manifest.permission.CAMERA)==PackageManager.PERMISSION_DENIED ||
                    checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)==PackageManager.PERMISSION_DENIED){
                // daca nu ai permisiunea fa request
                String[] permission={Manifest.permission.CAMERA,Manifest.permission.WRITE_EXTERNAL_STORAGE};

                requestPermissions(permission,PERMISSION_CODE);
            }
            else{
                //permisiunea e deja acceptat
                deschideCamera();
            }

        }
        else if (option == 1)
        {
            // Utilizatorul a ales galeria
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("image/*");

            startActivityForResult(intent, IMAGE_GALLERY_CODE);
        }
    }

    // Functie care se ocupa de onClick butoane
    private void handleButoane()
    {
        // Inapoi
        Button bBack = findViewById(R.id.button_back_clasif);
        bBack.setOnClickListener(view -> {
            finish();
        });

        // Galerie
        Button bGallery = findViewById(R.id.button_gallery_clasif);
        bGallery.setOnClickListener(view -> {
            Intent intent = new Intent(ClassifyActivity.this, ClassGalleryActivity.class);
            startActivity(intent);
        });

        // Anulare
        Button bCancel = findViewById(R.id.button_anulare_clasif);
        bCancel.setOnClickListener(view -> {
//            mImageView.setImageResource(android.R.color.transparent);
            mImageView.setImageDrawable(null);
            mCaptureBtn.setEnabled(true);
            mClassifierBtn.setEnabled(false);
            mZonaClasificare.setVisibility(View.INVISIBLE);
            mZonaActiuni.setVisibility(View.INVISIBLE);
        });

        // Adaugare TODO
        Button bAdd = findViewById(R.id.button_adaug_classif);
        bAdd.setOnClickListener(view -> {
            // Luam obiectul cu datele
            AppSavedData appSavedData = AppSavedData.load(this);
            Log.i("TEst", Integer.toString(appSavedData.poze.size()));
            // Construim obiectul cu datele
            ImageData imageData = new ImageData(path.toString(), Calendar.getInstance().getTime(), clasaGasita);
            appSavedData.poze.add(imageData);
            try {
                appSavedData.save(this);
            } catch (IOException e) {
                e.printStackTrace();
            }

            // Alert poza a fost adaugata
            new AlertDialog.Builder(this)
                    .setTitle("Succes")
                    .setMessage("Poza a fost adaugata cu succes")

                    // Specifying a listener allows you to take an action before dismissing the dialog.
                    // The dialog is automatically dismissed when a dialog button is clicked.
                    .setPositiveButton(android.R.string.yes, null)
//                    .setIcon() TODO
                    .show();

            // Intoarcem pagina la starea intiala
            clasaGasita = -1;
            mImageView.setImageDrawable(null);
            mCaptureBtn.setEnabled(true);
            mClassifierBtn.setEnabled(false);
            mZonaActiuni.setVisibility(View.INVISIBLE);
            mZonaClasificare.setVisibility(View.INVISIBLE);
        });

        // Handler click adaugare poza
        mCaptureBtn.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                // Afiseaza dialog cu optiuni
                ArrayChoiceDialog choiceDialog = new ArrayChoiceDialog(getString(R.string.inputChoice),new String[]{"Camera", "Galerie"});
                choiceDialog.show(getSupportFragmentManager(),null);
            }
        });

        mClassifierBtn.setOnClickListener(v -> {
            if(reteaNeuronala == null)
                return;
            Context context = mImageView.getContext();

            try {
                BitmapDrawable drawable = (BitmapDrawable) mImageView.getDrawable();
                Bitmap bitmap = drawable.getBitmap();

                // Preprocesarea datelor (setare dimensiune si normalizare)
                ImageProcessor imageProcessor =
                        new ImageProcessor.Builder()
                                .add(new ResizeOp(reteaNeuronala.getImgSize(), reteaNeuronala.getImgSize(), ResizeOp.ResizeMethod.BILINEAR))
                                .add(new NormalizeOp(0, 255))
                                .build();

                // Create a TensorImage object. This creates the tensor of the corresponding
                // tensor type (flot32 in this case) that the TensorFlow Lite interpreter needs.
                TensorImage tImage = new TensorImage(DataType.FLOAT32);

                // Analysis code for every frame
                // Preprocess the image
                tImage.load(bitmap);
                tImage = imageProcessor.process(tImage);

                // Creates inputs for reference.
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, reteaNeuronala.getImgSize(), reteaNeuronala.getImgSize(), 3}, DataType.FLOAT32);
                inputFeature0.loadBuffer(tImage.getBuffer());


                // Rezultatele in forma de vector de probabilitati
                float[] rezultate = reteaNeuronala.processImage(inputFeature0);
                // Sortam indicii ca sa gasim primele 3 clase
                ArrayIndexComparator comparator = new ArrayIndexComparator(rezultate);
                Integer[] indexes = comparator.createIndexArray();
                Arrays.sort(indexes, comparator);

                // Afisare nume clase
                mlistaClase.get(0).setText(String.format("%s", Constants.clase[indexes[0]]));
                mlistaClase.get(1).setText(String.format("%s", Constants.clase[indexes[1]]));
                mlistaClase.get(2).setText(String.format("%s", Constants.clase[indexes[2]]));

                // Afisare procentul claseor
                mlistaProcent.get(0).setText(String.format("%.1f%%", rezultate[indexes[0]]*100));
                mlistaProcent.get(1).setText(String.format("%.1f%%", rezultate[indexes[1]]*100));
                mlistaProcent.get(2).setText(String.format("%.1f%%", rezultate[indexes[2]]*100));

                // Afisam poze clase
                mlistaImagini.get(0).setImageDrawable(ContextCompat.getDrawable(context, Constants.poze[indexes[0]]));
                mlistaImagini.get(1).setImageDrawable(ContextCompat.getDrawable(context,Constants.poze[indexes[1]]));
                mlistaImagini.get(2).setImageDrawable(ContextCompat.getDrawable(context,Constants.poze[indexes[2]]));

                // Salvam datele necesare la add
                clasaGasita = indexes[0];

                // Log cu tot vectorul de probabilitati
                Log.i("RESULT", Arrays.toString(rezultate));
                Log.i("INDEXES", Arrays.toString(indexes));

                // Blocam butonul poza, daca vrea alta poza, va apasa pe anulare
                mCaptureBtn.setEnabled(false);

                // Afisam zona detalii
                mZonaClasificare.setVisibility(View.VISIBLE);

                // Afisam zona de actiuni
                mZonaActiuni.setVisibility(View.VISIBLE);

                // Ascunde buton clasifica
                mClassifierBtn.setEnabled(false);

            } catch (Exception e) {
                e.printStackTrace();
            }

//                return getSortedResult(result);

        });
    }
}

class ArrayIndexComparator implements Comparator<Integer>
{
    private final float[] array;

    public ArrayIndexComparator(float[] array)
    {
        this.array = array;
    }

    public Integer[] createIndexArray()
    {
        Integer[] indexes = new Integer[array.length];
        for (int i = 0; i < array.length; i++)
        {
            indexes[i] = i; // Autoboxing
        }
        return indexes;
    }

    @Override
    public int compare(Integer index1, Integer index2)
    {
        // Autounbox from Integer to int to use as array indexes
        return -Float.compare(array[index1],array[index2]);
    }


}