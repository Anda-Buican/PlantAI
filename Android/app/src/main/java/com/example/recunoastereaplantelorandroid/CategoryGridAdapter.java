package com.example.recunoastereaplantelorandroid;

import android.content.Context;
import android.graphics.Color;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.core.content.ContextCompat;

import com.example.recunoastereaplantelorandroid.utils.Constants;

import org.checkerframework.checker.signedness.qual.Constant;

import java.util.List;

public class CategoryGridAdapter extends BaseAdapter {

    private final Context context;
    private final List<PlantCategory> categories;

    public CategoryGridAdapter(Context context, List<PlantCategory> categories) {
        this.context = context;
        this.categories = categories;
    }

    @Override
    public int getCount() {
        return categories.size();
    }

    @Override
    public Object getItem(int i) {
        return categories.get(i);
    }

    @Override
    public long getItemId(int i) {
        return 0;
    }

    @Override
    public View getView(int position, View view, ViewGroup viewGroup) {
        LayoutInflater layoutInflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View root = layoutInflater.inflate(R.layout.category_square, viewGroup, false);
        root.setClipToOutline(true);
        PlantCategory categorie = categories.get(position);

        ImageView categoryPic = (ImageView) root.findViewById(R.id.imagineCategorieClasa);
        TextView titleText = (TextView) root.findViewById(R.id.titluCategorie);
        TextView numberText = (TextView) root.findViewById(R.id.numarPozeInCategorie);

        categoryPic.setImageDrawable(ContextCompat.getDrawable(context, Constants.poze[categorie.getCategoryId()]));
        titleText.setText(categorie.getTitle());
        numberText.setText(String.format("(%s)", categorie.getNrPictures()));

        // Setare background verde daca are cel putin o poza
        if(categorie.getNrPictures() > 0)
        {
            root.setBackground(ContextCompat.getDrawable(context, R.drawable.rounded_corners_bg3));
        }

        return root;
    }
}
