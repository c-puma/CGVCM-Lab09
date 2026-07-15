import cv2
from ultralytics import YOLO
import cvzone
import math
import time


# Cargar el modelo YOLOv8
modelo = YOLO("yolov8n.pt")

# Abrir la cámara
camara = cv2.VideoCapture(0)

camara.set(3, 1280)   # Ancho
camara.set(4, 720)    # Alto

tiempo_anterior = 0

# Bucle principal
while True:

    exito, imagen = camara.read()

    if not exito:
        print("No fue posible acceder a la cámara.")
        break

    # Detectar objetos
    resultados = modelo(imagen, stream=True)

    for resultado in resultados:

        cajas = resultado.boxes

        for caja in cajas:

            # Coordenadas
            x1, y1, x2, y2 = caja.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            ancho = x2 - x1
            alto = y2 - y1

            # Confianza
            confianza = math.ceil((caja.conf[0] * 100)) / 100

            # Clase
            clase = int(caja.cls[0])
            nombre = modelo.names[clase]

            # Solo mostrar objetos con buena confianza
            if confianza > 0.50:

                # Rectángulo bonito
                cvzone.cornerRect(
                    imagen,
                    (x1, y1, ancho, alto),
                    l=20,
                    rt=2,
                    colorR=(255, 0, 255)
                )

                texto = f"{nombre} {confianza*100:.1f}%"

                cvzone.putTextRect(
                    imagen,
                    texto,
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                    offset=5
                )

    # Calcular FPS
    tiempo_actual = time.time()
    fps = 1 / (tiempo_actual - tiempo_anterior)
    tiempo_anterior = tiempo_actual

    cv2.putText(
        imagen,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Mostrar ventana
    cv2.imshow("Reconocimiento y Clasificacion de Objetos", imagen)

    # Salir con Q
    tecla = cv2.waitKey(1)

    if tecla == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()