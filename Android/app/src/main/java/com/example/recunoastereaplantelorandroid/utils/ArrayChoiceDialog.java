package com.example.recunoastereaplantelorandroid.utils;

import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.DialogFragment;

import com.example.recunoastereaplantelorandroid.R;

public class ArrayChoiceDialog extends DialogFragment {

    private final String[] options;
    private final String title;

    public interface ArrayChoiceDialogListener {
        public void onDialogClick(DialogFragment dialog, int option);
    }

    // Use this instance of the interface to deliver action events
    ArrayChoiceDialogListener listener;

    public ArrayChoiceDialog(String title,String[] options)
    {
        this.options = options;
        this.title = title;
    }

    // Override the Fragment.onAttach() method to instantiate the NoticeDialogListener
    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        // Verify that the host activity implements the callback interface
        try {
            // Instantiate the NoticeDialogListener so we can send events to the host
            listener = (ArrayChoiceDialogListener) context;
        } catch (ClassCastException e) {
            // The activity doesn't implement the interface, throw exception
            throw new ClassCastException(context.toString()
                    + " must implement NoticeDialogListener");
        }
    }

    @NonNull
    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        builder.setTitle(title)
                .setItems(options, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int index) {
                        // index reprezinta pozitia pe care a apasat userul
                        listener.onDialogClick(ArrayChoiceDialog.this, index);
                    }
                });
        return builder.create();
    }
}
