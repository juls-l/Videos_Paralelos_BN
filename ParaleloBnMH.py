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

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bn = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

        out.write(frame_bn)

        frame_actual += 1
        cuadros_acumulados += 1

        if cuadros_acumulados >= 30:
            cola.put(cuadros_acumulados)
            cuadros_acumulados = 0

    if cuadros_acumulados > 0:
        cola.put(cuadros_acumulados)

    cap.release()
    out.release()


if __name__ == '__main__':
    ruta_entrada = "Video/video.webm"
    ruta_salida = "resultado/videobn_TodosHilos.mp4"

    num_hilos = mp.cpu_count()

    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print("No se pudo abrir el video de entrada. Verifica la ruta en Video/video.webm")
        sys.exit()

    fps = 60
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Resolución: {ancho}x{alto} a {fps} FPS")
    print(f"Total de fotogramas a procesar: {total_frames}")
    print(f"Hilos detectados en tu CPU: {num_hilos}")
    print(f"Iniciando procesamiento paralelo masivo ({num_hilos} hilos)...\n")

    cola = mp.Queue()
    frames_por_hilo = total_frames // num_hilos

    procesos = []
    rutas_temporales = []

    for i in range(num_hilos):
        inicio_f = i * frames_por_hilo
        fin_f = total_frames if i == (num_hilos - 1) else (i + 1) * frames_por_hilo

        ruta_temp = f"resultado/temp_multihilo_{i + 1}.mp4"
        rutas_temporales.append(ruta_temp)

        args = (i + 1, ruta_entrada, ruta_temp, inicio_f, fin_f, ancho, alto, fps, cola)
        p = mp.Process(target=procesar_bloque_video, args=args)
        procesos.append(p)

    inicio = time.time()

    for p in procesos:
        p.start()

    frames_completados = 0
    while frames_completados < total_frames:
        while not cola.empty():
            frames_completados += cola.get()

        if total_frames > 0:
            porcentaje = (frames_completados / total_frames) * 100
            sys.stdout.write(
                f"\rProcesando ({num_hilos} Hilos): {frames_completados}/{total_frames} frames ({porcentaje:.1f}%)")
            sys.stdout.flush()

        time.sleep(0.01)

        if all(not p.is_alive() for p in procesos) and cola.empty():
            break

    sys.stdout.write(f"\rProcesando ({num_hilos} Hilos): {total_frames}/{total_frames} frames (100.0%)\n\n")
    sys.stdout.flush()

    for p in procesos:
        p.join()

    print("Uniendo las partes temporales generadas...")
    out_final = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto), isColor=True)

    for ruta_temp in rutas_temporales:
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

    print("Procesamiento multi-hilo completado con éxito.")
    print(f"Tiempo total: {tiempo_total:.2f} segundos ({tiempo_total / 60:.2f} minutos)")
    print(f"Nuevo video guardado en: {ruta_salida}")