import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
import keras
from keras import layers, models
import cv2  
from skimage.measure import label, regionprops 
from scipy.spatial import distance 


base_path = r"C:\Users\elmou\OneDrive\Desktop\Master VMI FA\cours\imagbiomed\TP\TP3\DIC-C2DH-HeLa\DIC-C2DH-HeLa"

def load_data(path, seq='01'):
    img_paths = sorted(glob(os.path.join(path, seq, "*.tif")))
    mask_paths = sorted(glob(os.path.join(path, f"{seq}_ST", "SEG", "*.tif")))
    
    X, Y = [], []
    print(f"Chargement de {len(img_paths)} images depuis {path}...")
    
    for img_p, mask_p in zip(img_paths[:50], mask_paths[:50]):

        img = cv2.imread(img_p, -1)
        mask = cv2.imread(mask_p, -1)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
 
        img = (img - np.mean(img)) / np.std(img)
        mask = (mask > 0).astype(np.float32) 
        
        X.append(img)
        Y.append(mask)
    
    return np.array(X)[..., np.newaxis], np.array(Y)[..., np.newaxis]

try:

    X, Y = load_data(base_path)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
    print("Données prêtes.")
    

    def build_simple_unet():
        inputs = layers.Input((256, 256, 1))

        c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        p1 = layers.MaxPooling2D()(c1)
        c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
        p2 = layers.MaxPooling2D()(c2)

        c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)

        u4 = layers.UpSampling2D()(c3)
        c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u4)
        u5 = layers.UpSampling2D()(c4)
        c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u5)
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
        return models.Model(inputs, outputs)

    model = build_simple_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    print("Entraînement en cours (30 epochs)... Patientez environ 1 minute.")
    history = model.fit(X_train, Y_train, epochs=30, batch_size=8, validation_data=(X_val, Y_val))

    idx = 0 
    img_input = X_val[idx:idx+1]
    pred = model.predict(img_input)[0]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.title("Image DIC"); plt.imshow(X_val[idx,:,:,0], cmap='gray')
    plt.subplot(1, 3, 2); plt.title("Vérité (Ground Truth)"); plt.imshow(Y_val[idx,:,:,0], cmap='gray')
    plt.subplot(1, 3, 3); plt.title("Prédiction U-Net"); plt.imshow(pred[:,:,0], cmap='gray')
    plt.show()

    print("DÉMARRAGE DU TRACKING (SUIVI TEMPOREL)")
    print("="*40)

    img_t0 = X[0:1] 
    img_t1 = X[1:2] 


    mask_t0 = (model.predict(img_t0)[0, :, :, 0] > 0.5).astype(int)
    mask_t1 = (model.predict(img_t1)[0, :, :, 0] > 0.5).astype(int)


    props_t0 = regionprops(label(mask_t0))
    props_t1 = regionprops(label(mask_t1))

    print(f"Frame T=0 : {len(props_t0)} cellules détectées.")
    print(f"Frame T=1 : {len(props_t1)} cellules détectées.")
    print("-" * 40)

    links_found = 0
    
    for cell_t0 in props_t0:
        centroid_t0 = np.array(cell_t0.centroid)
        
        min_dist = 50.0 
        best_match_id = None

        for cell_t1 in props_t1:
            centroid_t1 = np.array(cell_t1.centroid)
            d = distance.euclidean(centroid_t0, centroid_t1)
            
            if d < min_dist:
                min_dist = d
                best_match_id = cell_t1.label
        

        if best_match_id is not None:
            print(f"[TRACK] Cellule {cell_t0.label} (t0) ---> Cellule {best_match_id} (t1) | Distance: {min_dist:.2f} px")
            links_found += 1
        else:
            print(f"[LOST]  Cellule {cell_t0.label} (t0) ---> Perdue (Trop loin ou disparue)")

    print("-" * 40)
    print(f"Tracking terminé. {links_found} liens établis sur {len(props_t0)} cellules.")

except Exception as e:
    print(f"\nUne erreur est survenue : {e}")