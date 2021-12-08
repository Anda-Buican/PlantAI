package com.example.recunoastereaplantelorandroid;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.text.format.DateFormat;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.core.content.ContextCompat;

import com.example.recunoastereaplantelorandroid.utils.Constants;

import java.io.File;
import java.net.URI;
import java.util.List;

public class PhotoGridAdapter extends BaseAdapter {

    private final Context context;
    private final List<ImageData> photos;

    public PhotoGridAdapter(Context context, List<ImageData> photos) {
        this.context = context;
        this.photos = photos;
    }

    @Override
    public int getCount() {
        return photos.size();
    }

    @Override
    public Object getItem(int i) {
        return photos.get(i);
    }

    @Override
    public long getItemId(int i) {
        return 0;
    }

    @Override
    public View getView(int position, View view, ViewGroup viewGroup) {
        LayoutInflater layoutInflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View root = layoutInflater.inflate(R.layout.photo_square, viewGroup, false);
        root.setClipToOutline(true);
        ImageData fotografie = photos.get(position);

        ImageView imageView = (ImageView) root.findViewById(R.id.imagine);
        TextView dataPozei = (TextView) root.findViewById(R.id.dataPozei);

        imageView.setImageURI(Uri.parse(fotografie.path));
        imageView.refreshDrawableState();
        String data = (String) DateFormat.format("dd-MM-yyyy hh:mm", fotografie.dataAdaugare);
        dataPozei.setText(data);

        return root;
    }
}
