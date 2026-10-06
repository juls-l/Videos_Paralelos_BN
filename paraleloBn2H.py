import cv2  # OpenCV: Para lectura y escritura de videos
import numpy as np  # NumPy: Manejo de matrices de imagen
import time  # time: Para cronometrar el tiempo de procesamiento
import sys  # sys: Para actualizar el porcentaje en la misma línea
import multiprocessing as mp  # multiprocessing: Módulo nativo para 2 núcleos reales
import os  # os: Para borrar los archivos temporales al finalizar


# ==========================================
# 1. FUNCIÓN QUE EJECUTA CADA NÚCLEO (PROCESO)
# ==========================================
def procesar_bloque_video(id_proceso, ruta_entrada, ruta_salida_temp, inicio_frame, fin_frame, ancho, alto, fps, cola):
    """
    Esta función se ejecuta de forma independiente en un núcleo de la CPU.
    Procesa únicamente el rango de fotogramas asignado: [inicio_frame, fin_frame).
    """
    cap = cv2.VideoCapture(ruta_entrada)
    # Posiciona el reproductor directamente en el fotograma que le toca a este núcleo
    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida_temp, fourcc, fps, (ancho, alto), isColor=True)

    frame_actual = inicio_frame
    cuadros_acumulados = 0

    while frame_actual < fin_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # --- CONVERSIÓN ULTRA RÁPIDA A BLANCO Y NEGRO ---
        # Paso 1: Convertir de BGR (color) a Escala de Grises (1 canal)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Paso 2: Reconvertir a 3 canales BGR para que VideoWriter lo acepte
        frame_bn = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

        # Guardar fotograma en el archivo temporal de este núcleo
        out.write(frame_bn)

        frame_actual += 1
        cuadros_acumulados += 1

        # NOTIFICACIÓN EN BLOQUES DE 30 FRAMES:
        # En lugar de avisar frame por frame (lo que ralentiza todo),
        # notifica a la cola cada 30 cuadros para eliminar la sobrecarga de comunicación.
        if cuadros_acumulados >= 30:
            cola.put(cuadros_acumulados)
            cuadros_acumulados = 0

    # Enviar los cuadros restantes al terminar la lectura
    if cuadros_acumulados > 0:
        cola.put(cuadros_acumulados)

    cap.release()
    out.release()


# ==========================================
# 2. BLOQUE PRINCIPAL (ORQUESTADOR EN PYTHON)
# ==========================================
if __name__ == '__main__':
    # --- A. RUTAS DE ENTRADA, SALIDA Y TEMPORALES ---
    ruta_entrada = "Video/video.webm"
    ruta_salida = "resultado/videobn_paralelo.mp4"
    temp_parte1 = "resultado/temp_parte1.mp4"
    temp_parte2 = "resultado/temp_parte2.mp4"

    # --- B. LEER METADATOS DEL VIDEO ---
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print("No se pudo abrir el video de entrada. Verifica la ruta.")
        sys.exit()

    fps = 60  # Forzado a 60 FPS
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ancho del video
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Alto del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Fotogramas totales
    cap.release()  # Liberar para los subprocesos

    print(f"Resolución: {ancho}x{alto} a {fps} FPS")
    print(f"Total de fotogramas a procesar: {total_frames}")
    print("Iniciando procesamiento paralelo con 2 núcleos...\n")

    # --- C. DIVIDIR EL TRABAJO Y CREAR RECURSOS COMPARTIDOS ---
    cola = mp.Queue()  # "Buzón" de comunicación entre procesos
    mitad_frames = total_frames // 2  # Punto medio para repartir 50% y 50%

    # Argumentos para el Núcleo 1 (Primera mitad: 0 a mitad)
    args_p1 = (1, ruta_entrada, temp_parte1, 0, mitad_frames, ancho, alto, fps, cola)
    # Argumentos para el Núcleo 2 (Segunda mitad: mitad a final)
    args_p2 = (2, ruta_entrada, temp_parte2, mitad_frames, total_frames, ancho, alto, fps, cola)

    # --- D. ARRANCAR LOS 2 PROCESOS EN PARALELO ---
    inicio = time.time()  # Iniciar cronómetro

    p1 = mp.Process(target=procesar_bloque_video, args=args_p1)
    p2 = mp.Process(target=procesar_bloque_video, args=args_p2)

    p1.start()
    p2.start()

    # --- E. MONITOREO DEL PORCENTAJE EN VIVO (\r) ---
    frames_completados = 0
    while frames_completados < total_frames:
        # Revisa el buzón (cola) y suma los avances reportados por ambos núcleos
        while not cola.empty():
            frames_completados += cola.get()

        if total_frames > 0:
            porcentaje = (frames_completados / total_frames) * 100
            # \r regresa el cursor al inicio para sobrescribir la misma línea en pantalla
            sys.stdout.write(
                f"\rProcesando (2 Núcleos): {frames_completados}/{total_frames} frames ({porcentaje:.1f}%)")
            sys.stdout.flush()

        time.sleep(0.01)

        # Si ambos núcleos ya terminaron y no hay datos en la cola, rompe el ciclo
        if not p1.is_alive() and not p2.is_alive() and cola.empty():
            break

    # Imprimir 100% final al salir
    sys.stdout.write(f"\rProcesando (2 Núcleos): {total_frames}/{total_frames} frames (100.0%)\n\n")
    sys.stdout.flush()

    # Esperar a que los procesos cierren limpiamente
    p1.join()
    p2.join()

    # --- F. UNIR LOS VIDEOS TEMPORALES EN EL DEFINITIVO ---
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
        os.remove(ruta_temp)  # Eliminar archivo temporal automáticamente

    out_final.release()

    # --- G. TIEMPO FINAL ---
    fin = time.time()
    tiempo_total = fin - inicio

    print("Procesamiento paralelo finalizado exitosamente.")
    print(f"Tiempo total: {tiempo_total:.2f} segundos ({tiempo_total / 60:.2f} minutos)")
    print(f"Video guardado en: {ruta_salida}")