from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("best_completo.pt")  # Cambia "best_Exp.pt"
#bestAumentado.pt funciona bien con 80 de confianza es el que fue entrenado sin aumentar nada solo el dataset con las fotos que metimos de armando
#el completo, el best_exp y el best funcionan bien
#el completo es el que fue entrenado con las nuevas que envio armndo y fue aumentada tanto el de pruebas como el de train
#best aumetnado fue el entrenamiento aumentado, sin aumentar el dataset con las imagenes de Armando 
#best_Exp es el que mejor funcionaba antes 
# Iniciar la captura de video desde la webcam
cap = cv2.VideoCapture(1)  # 0 indica la webcam predeterminada

# Bucle para procesar la entrada de la webcam
while True:
    # Capturar fotograma
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    # Realizar la predicción con YOLOv8 (filtro por confianza mayor o igual a 90%)
    resultados = model.predict(frame, imgsz=640, verbose=False, conf=0.80) 

    # Verificar si hay detecciones válidas
    if len(resultados[0].boxes) > 0:  # Si hay predicciones
        anotaciones = resultados[0].plot()  # Dibujar las cajas con anotaciones
    else:
        anotaciones = frame  # Si no hay detecciones, mostrar el fotograma original

    # Mostrar el fotograma con las detecciones
    cv2.imshow("Detección con YOLOv8 (Confianza > 90%)", anotaciones)

    # Salir del programa si se presiona la tecla 'ESC' (código 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
