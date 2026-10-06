import cv2  # OpenCV: Para lectura y escritura de videos
import numpy as np  # NumPy: Manejo de matrices de imagen
import time  # time: Medición de tiempos
import sys  # sys: Para la barra en una sola línea (\r)
import multiprocessing as mp  # multiprocessing: Detección y uso de todos los hilos
import os  # os: Limpieza de archivos temporales


# ==========================================
# 1. FUNCIÓN QUE EJECUTA CADA HILO / NÚCLEO
# ==========================================
def procesar_bloque_video(id_proceso, ruta_entrada, ruta_salida_temp, inicio_frame, fin_frame, ancho, alto, fps, cola):
    """
    Función que ejecuta cada hilo independiente de la CPU.
    Procesa un rango específico de fotogramas [inicio_frame, fin_frame).
    """
    cap = cv2.VideoCapture(ruta_entrada)
    # Posicionar el lector en el fotograma inicial asignado a este hilo
    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida_temp, fourcc, fps, (ancho, alto), isColor=True)

    frame_actual = inicio_frame
    cuadros_acumulados = 0

    while frame_actual < fin_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # --- CONVERSIÓN RÁPIDA EN OPENCV NATIVO ---
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bn = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

        out.write(frame_bn)

        frame_actual += 1
        cuadros_acumulados += 1

        # Notificar a la cola en bloques de 30 cuadros para eliminar la sobrecarga
        if cuadros_acumulados >= 30:
            cola.put(cuadros_acumulados)
            cuadros_acumulados = 0

    # Enviar remanente de cuadros al finalizar la lectura
    if cuadros_acumulados > 0:
        cola.put(cuadros_acumulados)

    cap.release()
    out.release()


# ==========================================
# 2. BLOQUE PRINCIPAL (ORQUESTADOR MULTI-HILO)
# ==========================================
if __name__ == '__main__':
    # --- A. RUTAS CON NOMBRES ÚNICOS PARA NO SOBREESCRIBIR LO ANTERIOR ---
    ruta_entrada = "Video/video.webm"
    ruta_salida = "resultado/videobn_TodosHilos.mp4"

    # Detectar automáticamente el número total de hilos de tu CPU
    num_hilos = mp.cpu_count()

    # --- B. LEER METADATOS DEL VIDEO ORIGINAL ---
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

    # --- C. CALCULAR DIVISIONES DE PANTALLA Y RANGOS ---
    cola = mp.Queue()
    frames_por_hilo = total_frames // num_hilos

    procesos = []
    rutas_temporales = []

    # Configuración dinámica para cada hilo
    for i in range(num_hilos):
        inicio_f = i * frames_por_hilo
        # El último hilo absorbe cualquier fotograma sobrante por división
        fin_f = total_frames if i == (num_hilos - 1) else (i + 1) * frames_por_hilo

        # Archivos temporales con prefijo único
        ruta_temp = f"resultado/temp_multihilo_{i + 1}.mp4"
        rutas_temporales.append(ruta_temp)

        args = (i + 1, ruta_entrada, ruta_temp, inicio_f, fin_f, ancho, alto, fps, cola)
        p = mp.Process(target=procesar_bloque_video, args=args)
        procesos.append(p)

    # --- D. ARRANCAR TODOS LOS PROCESOS SIMULTÁNEAMENTE ---
    inicio = time.time()

    for p in procesos:
        p.start()

    # --- E. MONITOREO DE PROGRESO GLOBAL (\r) ---
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

        # Romper bucle si todos los procesos terminaron
        if all(not p.is_alive() for p in procesos) and cola.empty():
            break

    # Asegurar impresión del 100% final
    sys.stdout.write(f"\rProcesando ({num_hilos} Hilos): {total_frames}/{total_frames} frames (100.0%)\n\n")
    sys.stdout.flush()

    for p in procesos:
        p.join()

    # --- F. UNIR LAS N PARTES TEMPORALES ---
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
        os.remove(ruta_temp)  # Limpieza de temporales al vuelo

    out_final.release()

    # --- G. TIEMPO FINAL Y RESUMEN ---
    fin = time.time()
    tiempo_total = fin - inicio

    print("Procesamiento multi-hilo completado con éxito.")
    print(f"Tiempo total: {tiempo_total:.2f} segundos ({tiempo_total / 60:.2f} minutos)")
    print(f"Nuevo video guardado en: {ruta_salida}")