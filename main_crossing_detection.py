'''
Этот код предназначен для обработки видеозаписей, в которых требуется выявить нарушения ПДД, такие как пересечение линии или нарушение стоп-линии. Он использует компьютерное зрение для обработки каждого кадра видео и выявления транспортных средств, пересекающих определенные границы.

Описание основных частей кода:
Функции:
   - `img_detect_color(roi)`: Функция для фильтрации цвета и выделения объектов, используя маску для белых и желтых цветов (например, для выделения дорожных разметок).
   - `is_duplicate_violation(x, y, threshold=20)`: Проверяет, является ли текущее нарушение дублирующимся, чтобы избежать повторной регистрации нарушения при незначительных смещениях.
   - `log_violation(video_name, timestamp, violation_type)`: Записывает информацию о нарушении в CSV файл, включая название видео, время нарушения и тип нарушения.
   - `smooth_detection(detections, window_size=5)`: Сглаживает координаты детекций с помощью скользящего среднего, чтобы уменьшить влияние случайных ошибок.
   - `normalize_detection(x, y, frame_width, frame_height)`: Нормализует координаты в диапазоне от 0 до 1 относительно ширины и высоты кадра.
   - `process_frame(frame, video_name, current_time)`: Основная функция обработки каждого кадра, которая:
     - Создает трапециевидную область интереса (ROI) для отслеживания движения объектов.
     - Применяет цветовую фильтрацию для выделения объектов.
     - Обрабатывает линии, найденные с помощью преобразования Хафа.
     - Проверяет, пересек ли объект линию или нарушил стоп-линию.
     - Регистрирует нарушения и отображает информацию о нарушениях на изображении.
   - `process_video(video_path)`: Основная функция обработки видеофайла. Она загружает видео, обрабатывает кадры и вызывает функцию для детекции нарушений.

Прочие особенности:
   - Время записи нарушений в CSV файл форматируется как `чч:мм:сс`.
   - Название CSV файла включает название видео и дату обработки.
   - Каждое нарушение фиксируется в CSV файле с указанием типа нарушения, времени и названия видео.

Применение:
Этот код предназначен для анализа видеозаписей с целью выявления нарушений дорожного движения, таких как пересечение линий или нарушение стоп-линий, и последующей регистрации нарушений в CSV файл для дальнейшего анализа.
'''

import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Инициализируем счетчики нарушений
crossing_counter = 0
stop_line_violation_counter = 0
detection_buffer = []  # Буфер для хранения предыдущих детекций
detection_history = []  # История детекций для усреднения

def img_detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    return blurred

def is_duplicate_violation(x, y, threshold=20):
    """Проверка на наличие уже зарегистрированного нарушения в буфере."""
    for (px, py) in detection_buffer:
        if abs(px - x) < threshold and abs(py - y) < threshold:
            return True
    return False

def log_violation(video_name, timestamp, violation_type):
    """Записывает нарушение в CSV файл."""
    # Форматирование времени в чч:мм:сс
    formatted_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

    # Формирование названия CSV файла с датой и названием видео
    date_str = datetime.now().strftime('%Y-%m-%d')
    csv_filename = f"violations_log_{video_name}_{date_str}.csv"

    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # Записываем заголовки, если файл не существует
            writer.writerow(["Video Name", "Timestamp (hh:mm:ss)", "Violation Type"])
        writer.writerow([video_name, formatted_time, violation_type])

def smooth_detection(detections, window_size=5):
    """Сглаживание координат с помощью скользящего среднего."""
    if len(detections) < window_size:
        return detections[-1]
    else:
        smoothed_x = np.mean([d[0] for d in detections[-window_size:]])
        smoothed_y = np.mean([d[1] for d in detections[-window_size:]])
        return (smoothed_x, smoothed_y)

def normalize_detection(x, y, frame_width, frame_height):
    """Нормализация координат в диапазоне [0, 1]."""
    normalized_x = x / frame_width
    normalized_y = y / frame_height
    return normalized_x, normalized_y

def process_frame(frame, video_name, current_time):
    global crossing_counter, stop_line_violation_counter
    height, width = frame.shape[:2]
    
    # Определяем координаты для трапециевидного ROI
    x_start_top = width // 2.4
    x_end_top = width - width // 2.5
    y_start = height // 2.5
    x_start_bottom = width * 1.5 // 8
    x_end_bottom = width * 4.5 // 5
    y_end = height - height // 4.7  

    # Маска для выделения области трапеции
    mask = np.zeros_like(frame)
    trapezoid_points = np.array([[x_start_top, y_start], [x_end_top, y_start],
                                 [x_end_bottom, y_end], [x_start_bottom, y_end]], np.float32)
    cv2.fillPoly(mask, [trapezoid_points.astype(int)], (255, 255, 255))

    roi = cv2.bitwise_and(frame, mask)
    filtered_mask = img_detect_color(roi)
    edges = cv2.Canny(filtered_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20)

    debug_frame = frame.copy()
    cv2.polylines(debug_frame, [trapezoid_points.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if not (-10 < angle < 10 or 170 < abs(angle) < 190):
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center_x = float(x1 + x2) / 2.0
                center_y = float(y1 + y2) / 2.0

                if cv2.pointPolygonTest(trapezoid_points, (center_x, center_y), False) >= 0:
                    if not is_duplicate_violation(center_x, center_y):
                        detection_history.append((center_x, center_y))
                        smoothed_detection = smooth_detection(detection_history)
                        normalized_x, normalized_y = normalize_detection(smoothed_detection[0], smoothed_detection[1], width, height)
                        crossing_counter += 1
                        detection_buffer.append((center_x, center_y))
                        log_violation(video_name, current_time, "Line Crossing")

            if y1 == y2 and cv2.pointPolygonTest(trapezoid_points, (float(x1 + x2) / 2.0, float(y1)), False) >= 0:
                if not is_duplicate_violation(x1, y1):
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    stop_line_violation_counter += 1
                    detection_buffer.append((x1, y1))
                    log_violation(video_name, current_time, "Stop Line Violation")

    if len(detection_buffer) > 10:
        detection_buffer.pop(0)

    cv2.polylines(frame, [trapezoid_points.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, f"Crossings: {crossing_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Stop Line Violations: {stop_line_violation_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Debug Frame", debug_frame)

    return frame

def process_video(video_path):
    global crossing_counter, stop_line_violation_counter
    crossing_counter = 0
    stop_line_violation_counter = 0
    detection_buffer.clear()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка открытия видеофайла")
        return

    video_name = os.path.basename(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        processed_frame = process_frame(frame, video_name, current_time)

        cv2.imshow("Lane Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Запуск обработки видео
video_path = "E:/ASUS_ROG/Работа/Хакатон/train РЖД ПДД/videos/AKN00048.mp4"
process_video(video_path)
