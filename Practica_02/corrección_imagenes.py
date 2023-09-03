import numpy as np
import cv2

#Cargar y mostrar Imagen Proyectada:
imagen_proyectada = cv2.imread('Ajedrez_Proyectado.jpg')
#cv2.imshow('Imagen Ortogonal', imagen_proyectada)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Cargar y mostrar Imagen Ortogonal:
imagen_modelo = cv2.imread('Ajedrez_Ortogonal.jpg')
#cv2.imshow('Imagen Ortogonal', imagen_proyectada)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Puntos de referencia de la imagen modelo (Ajedres_Ortogonal.jpg)
puntos_model = np.array([
    [140, 85],
    [140, 291],
    [140, 500],
    [555, 500],
    [555, 291],
    [555, 85],
    [347, 85],
    [347, 500],
    [347, 291]
])

# Puntos de referencia de la imagen a corregir (Ajedrez_Proyectado.jpg)
puntos_proy = np.array([
    [152, 267],
    [291, 333],
    [469, 423],
    [634, 221],
    [486, 177],
    [367, 141],
    [273, 192],
    [566, 302],
    [405, 243]
])

#Crear la matriz de coeficientes para resolver el sistema de ecuaciones
A = np.empty((0, 9))
for i in range(9):
    row1 = [-puntos_proy[i, 0], -puntos_proy[i, 1], -1, 0, 0, 0,
            puntos_model[i, 0] * puntos_proy[i, 0], puntos_model[i, 0] * puntos_proy[i, 1], puntos_model[i, 0]]
    row2 = [0, 0, 0, -puntos_proy[i, 0], -puntos_proy[i, 1], -1,
            puntos_model[i, 1] * puntos_proy[i, 0], puntos_model[i, 1] * puntos_proy[i, 1], puntos_model[i, 1]]
    A = np.vstack((A, row1, row2))

# Resuelve el sistema de ecuaciones homogéneo A*h=0
_, _, V = np.linalg.svd(A)
solution = V[-1, :]  # El último vector propio de V

H = np.reshape(solution, (3, 3))

# Normaliza la solución
H = H / H[2, 2]

print("H = ")
print(H)

# Aplicar la transformación perspectiva utilizando la matriz H
imagen_corregida = cv2.warpPerspective(imagen_proyectada, H, (imagen_proyectada.shape[1], imagen_proyectada.shape[0]))

# Guardar la imagen ortogonal resultante
cv2.imwrite('Imagen_corregida.jpg', imagen_corregida)



alpha = 0.5

imagen_modelo = cv2.resize(imagen_modelo, (imagen_proyectada.shape[1], imagen_proyectada.shape[0]))

overlapped_images = cv2.addWeighted(imagen_corregida, alpha, imagen_modelo, 1 - alpha, 0)


cv2.imwrite('Overlapped_images.jpg', overlapped_images)