package com.example.recunoastereaplantelorandroid;

public class PlantCategory {
    private String title;
    private int nrPictures, categoryId;

    public PlantCategory(int categoryId, String title, int nrPictures) {
        this.categoryId = categoryId;
        this.title = title;
        this.nrPictures = nrPictures;
    }

    public String getTitle() {
        return title;
    }

    public int getNrPictures() {
        return nrPictures;
    }

    public int getCategoryId() {
        return categoryId;
    }
}
