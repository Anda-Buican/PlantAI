import time

import joblib
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Variabile generale
DIM_POZA = 75
CANALE = 3
CACHE = True
CACHE_LOCATIE = 'Cache/cache'

print(f"Dim poza: {DIM_POZA}")

if not CACHE or not os.path.isfile(CACHE_LOCATIE+'_flat_'+str(DIM_POZA)+'.npy') or not os.path.isfile(CACHE_LOCATIE+'_target_'+str(DIM_POZA)+'.npy'):
    # categoriile din flowers
    categorii = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    flat_data_arr = []  # pozele ca vectori, nu matrice
    target_arr = []  # clasele pozelor

    # calea catre folderul cu poze
    datadir = 'flowers/'

    # incarcam fiecare poza categorie cu categorie
    for i in categorii:

        print(f'se incarca categoria: {i}')
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (DIM_POZA, DIM_POZA, CANALE))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(categorii.index(i))
        print(f's-a terminat categoria:{i}')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    np.save(CACHE_LOCATIE+'_flat_'+str(DIM_POZA),flat_data)
    np.save(CACHE_LOCATIE+'_target_'+str(DIM_POZA),target)
else:
    flat_data = np.load(CACHE_LOCATIE+'_flat_'+str(DIM_POZA)+'.npy')
    target = np.load(CACHE_LOCATIE+'_target_'+str(DIM_POZA)+'.npy')

print("S-au incarcat datele de intrare!")

df = pd.DataFrame(flat_data)  # dataframe
df['Target'] = target
x = df.iloc[:, :-1]  # input data
y = df.iloc[:, -1]  # output data

# Parametrii pe care ii vom incerca
# param_grid={'C':[1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf']}
# print(f"Parametrii: {param_grid}")

# construim SVM-ul
model=svm.SVC(C=1,gamma=0.001,kernel='rbf',probability=True,verbose=4)
# model=BaggingClassifier(svm.SVC(C=1,gamma=0.001,kernel='rbf',probability=True), max_samples=1.0 / N_ESTIMATORS, n_estimators=N_ESTIMATORS,n_jobs=2,verbose=4)
# model=svm.SVC(probability=True)

# Incercam diferite configuratii
# model=GridSearchCV(model, param_grid, verbose=4, n_jobs=6)

# Impartim pozele in test si train
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('----Am facut split----')
start = time.time()
model.fit(x_train,y_train)
done = time.time()
elapsed = done - start

# Afisare timp trecut
print("Durata antrenare")
print(str(int(elapsed/60)) + ":"+ str(int(elapsed)%60))

print('----S-a terminat train-ul----')
print('Cei mai buni parametrii:')
# print(model.best_params_)

# facem predictiile pe setul de testare
y_pred=model.predict(x_test)
print("Rezultate prezicere :")
print(y_pred)
print("Rezultate actuale:")
print(np.array(y_test))
# Calcul acuratete
print(f"Acuratetea este: {sklearn.metrics.accuracy_score(y_pred,y_test)*100}% ")

# SAalvam modelul pe disk
# joblib.dump(model.best_estimator_,'SkLearnModels/svm'+str(DIM_POZA)+'.joblib')
