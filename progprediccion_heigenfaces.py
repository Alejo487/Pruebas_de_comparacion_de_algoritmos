# coding=utf-8
# Importamos las librerías necesarias.
import os
import time
from argparse import ArgumentParser

import cv2
import imutils
import numpy as np
from imutils import paths, build_montages
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
def detect_faces(network, image, min_confidence=0.5):
    """
    Detecta rostros en una imagen utilizando una red neuronal.
    :param network: Instancia de una red neuronal entrenada en Caffe.
    :param image: Image sobre la que se llevarán a cabo las detecciones.
    :param min_confidence: Probabilidad mínima que debe tener una detección para no considerarse un falso positivo.
    """

    # Extraemos las dimensiones de la imagen.
    height, width = image.shape[:2]

    # Convertimos la imagen en un blob.
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 117.0, 123.0))

    # Pasamos la imagen por la red para obtener las detecciones.
    network.setInput(blob)
    detections = network.forward()

    # Iteramos sobre cada detección...
    boxes = []
    for i in range(0, detections.shape[2]):
        # Extraemos la confianza o probabilidad de la detección.
        confidence = detections[0, 0, i, 2]

        # Si la probabilidad es mayor que la mínima, guardamos las coordenadas de la detección y
        # la añadimos al resultado.
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x_start, y_start, x_end, y_end = box.astype('int')

            boxes.append((x_start, y_start, x_end, y_end))

    return boxes
    def load_face_dataset(input_path, network, min_confidence=0.5, min_samples=15):
    
    # Extraemos los nombres a partir de la carpeta contenedora de cada grupo de imágenes.
    image_paths = list(paths.list_images(input_path))
    names = [path.split(os.path.sep)[-2] for path in image_paths]

    # Contamos el número de imágenes por nombre.
    names, counts = np.unique(names, return_counts=True)
    names = names.tolist()

    faces = []
    labels = []

    # Iteramos sobre cada imagen...
    for image_path in image_paths:
        # Leemos la imagen y de la ruta de la misma extraemos el nombre de la persona a la que pertenece.
        image = cv2.imread(image_path)
        name = image_path.split(os.path.sep)[-2]

        # Si el nombre corresponde a un grupo muy pequeño, descartamos la imagen.
        if counts[names.index(name)] < min_samples:
            continue

        # Detectamos el rostro en la imagen.
        boxes = detect_faces(network, image, min_confidence)

        # Iteramos sobre las detecciones...
        for (x_start, y_start, x_end, y_end) in boxes:
            # Extraemos las regiones de los rostros en la imagen
            face_roi = image[y_start:y_end, x_start:x_end]
            face_roi = cv2.resize(face_roi, (47, 62))
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Añadimos la región de interés, junto con el nombre de la persona, al conjunto de datos.
            faces.append(face_roi)
            labels.append(name)

    faces = np.array(faces)
    labels = np.array(labels)

    # Retornamos los resultados
    return faces, labels
    # Definimos los parámetros del programa.
argument_parser = ArgumentParser()
argument_parser.add_argument('-i', '--input', type=str, required=True,
                             help='Ruta al directorio de entrada con las imágenes de CALTECH')
argument_parser.add_argument('-f', '--face', type=str, default='resources',
                             help='Ruta al directorio del detector de rostros')
argument_parser.add_argument('-c', '--confidence', type=float, default=0.5,
                             help='Mínima probabilidad para filtras detecciones débiles')
argument_parser.add_argument('-n', '--num-components', type=int, default=150,
                             help='Número de componentes principales (PCA).')
argument_parser.add_argument('-v', '--visualize', type=int, default=-1,
                             help='Flag para indicar si las componentes de PCA deberían ser visualizadas.')
argument_parser.add_argument('-s', '--sample', type=int, default=5, help='Número mínimo de imágenes por rostro.')
arguments = vars(argument_parser.parse_args())
# Instanciamos la red detectora de rostros.
print('Cargando el detector de rostros...')
prototxt_path = os.path.sep.join([arguments['face'], 'deploy.prototxt'])
weights_path = os.path.sep.join([arguments['face'], 'res10_300x300_ssd_iter_140000.caffemodel'])
network = cv2.dnn.readNet(prototxt_path, weights_path)
# Cargamos el conjunto de datos.
print('Cargando el conjunto de datos...')
faces, labels = load_face_dataset(arguments['input'], network, min_confidence=arguments['confidence'],
                                  min_samples=arguments['sample'])
print(f'Hay {len(faces)} imágenes en el conjunto de datos.')
# Aplana todos los rostros en 2D, convirtiéndolos en un vector unidimensional de píxeles.
pca_faces = np.array([face.flatten() for face in faces])
# Convertimos las etiquetas (los nombres de cada persona en el conjunto de datos) en números enteros.
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
# 75% de los datos para entrenamiento, 25% para pruebas.
original_train, original_test, X_train, X_test, y_train, y_test = train_test_split(faces, pca_faces, labels,
                                                                                   test_size=0.25,
                                                                                   stratify=labels,
                                                                                   random_state=9)
# Instanciamos el PCA.
print('Creando los Eigenfaces...')
pca = PCA(svd_solver='randomized', n_components=arguments['num_components'], whiten=True)

# Corremos PCA.
start = time.time()
X_train = pca.fit_transform(X_train)
print(f'La creación de Eigenfaces demoró {time.time() - start:.4f} segundos.')
# Validamos si tenemos que visualizar los Eigenfaces o no
if arguments['visualize'] > 0:
    images = []

    # Tomamos los primeros 16 componentes.
    for i, component in enumerate(pca.components_[:16]):
        component = component.reshape((62, 47))
        component = rescale_intensity(component, out_range=(0, 255))
        component = np.dstack([component.astype('uint8')] * 3)
        images.append(component)

    # Creamos un montaje de los componentes a visualizar.
    montage = build_montages(images, (47, 62), (4, 4))[0]

    # Obtenemos y procesamos la media del modelo PCA.
    mean = pca.mean_.reshape((62, 47))
    mean = rescale_intensity(mean, out_range=(0, 255)).astype('uint8')

    # Mostramos la media y los componentes.
    cv2.imshow('Media', mean)
    cv2.imshow('Componentes', montage)
    cv2.waitKey(0)
    # Entrenamos el clasificador.
print('Entrenando el clasificador...')
model = SVC(kernel='rbf', C=10.0, gamma=1e-3, random_state=9)
model.fit(X_train, y_train)
# Evaluamos el modelo, imprimiendo un reporte de clasificación.
print('Evaluando el clasificador...')
predictions = model.predict(pca.transform(X_test))
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))
# Seleccionamos una muestra de índices del conjunto de pruebas.
índices = np.random.choice(range(0, len(y_test)), size=10, replace=True)

para i en índices:
    # Obtenemos la predicción y el nombre real de la persona.
    nombre_predicho = codificador_etiqueta.transformación_inversa([predicciones[i]])[0]
    nombre_actual = codificador_etiqueta.clases_[y_test[i]]

    # Reformateamos la imagen.
    cara = np.dstack([prueba_original[i]] * 3)
    cara = imutils.resize(cara, ancho=250)

    # Imprimimos la predicción y el nombre real en la imagen.
    cv2.putText(cara, f'Pred: {nombre_predicho}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(cara, f'Real: {nombre_real}', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Imprimimos la predicción y el nombre real en la consola.
    print(f'Predicción: {predicted_name}, real: {actual_name} '
          f'[{"CORRECTA" if nombre_predicho == nombre_actual else "INCORRECTA"}]')

    # Mostramos el resultado.
    cv2.imshow('Rostro', cara)
    cv2.esperaClave(0)