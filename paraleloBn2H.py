import cv2
import numpy as np
import time
import sys
import multiprocessing as mp
import os


def procesar_bloque_video(id_proceso, ruta_entrada, ruta_salida_temp, inicio_frame, fin_frame, ancho, alto, fps, cola):
    cap = cv2.VideoCapture(ruta_entrada)
    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida_temp, fourcc, fps, (ancho, alto), isColor=True)

    frame_actual = inicio_frame
    cuadros_acumulados = 0

    while frame_actual < fin_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Conversión a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bn = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

        out.write(frame_bn)

        frame_actual += 1
        cuadros_acumulados += 1

        # Notifica a la cola cada 30 cuadros para reducir sobrecarga
        if cuadros_acumulados >= 30:
            cola.put(cuadros_acumulados)
            cuadros_acumulados = 0

    if cuadros_acumulados > 0:
        cola.put(cuadros_acumulados)

    cap.release()
    out.release()


if __name__ == '__main__':
    # Configuración de rutas
    ruta_entrada = "Video/video.webm"
    ruta_salida = "resultado/videobn_paralelo.mp4"
    temp_parte1 = "resultado/temp_parte1.mp4"
    temp_parte2 = "resultado/temp_parte2.mp4"

    # Lectura de metadatos
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print("No se pudo abrir el video de entrada. Verifica la ruta.")
        sys.exit()

    fps = 60
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Resolución: {ancho}x{alto} a {fps} FPS")
    print(f"Total de fotogramas a procesar: {total_frames}")
    print("Iniciando procesamiento paralelo con 2 núcleos...\n")

    # División del trabajo (50% cada núcleo)
    cola = mp.Queue()
    mitad_frames = total_frames // 2

    args_p1 = (1, ruta_entrada, temp_parte1, 0, mitad_frames, ancho, alto, fps, cola)
    args_p2 = (2, ruta_entrada, temp_parte2, mitad_frames, total_frames, ancho, alto, fps, cola)

    inicio = time.time()
    p1 = mp.Process(target=procesar_bloque_video, args=args_p1)
    p2 = mp.Process(target=procesar_bloque_video, args=args_p2)

    p1.start()
    p2.start()

    # Monitoreo del progreso en consola
    frames_completados = 0
    while frames_completados < total_frames:
        while not cola.empty():
            frames_completados += cola.get()

        if total_frames > 0:
            porcentaje = (frames_completados / total_frames) * 100
            sys.stdout.write(
                f"\rProcesando (2 Núcleos): {frames_completados}/{total_frames} frames ({porcentaje:.1f}%)")
            sys.stdout.flush()

        time.sleep(0.01)

        if not p1.is_alive() and not p2.is_alive() and cola.empty():
            break

    sys.stdout.write(f"\rProcesando (2 Núcleos): {total_frames}/{total_frames} frames (100.0%)\n\n")
    sys.stdout.flush()

    p1.join()
    p2.join()

    # Unión de archivos temporales y limpieza
    print("Uniendo partes temporales...")
    out_final = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto), isColor=True)

    for ruta_temp in [temp_parte1, temp_parte2]:
        cap_temp = cv2.VideoCapture(ruta_temp)
        while True:
            ret, frame = cap_temp.read()
            if not ret:
                break
            out_final.write(frame)
        cap_temp.release()
        os.remove(ruta_temp)

    out_final.release()

    fin = time.time()
    tiempo_total = fin - inicio

    print("Procesamiento paralelo finalizado exitosamente.")
    print(f"Tiempo total: {tiempo_total:.2f} segundos ({tiempo_total / 60:.2f} minutos)")
    print(f"Video guardado en: {ruta_salida}")