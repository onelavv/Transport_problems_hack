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




В данном коде применяются различные научные принципы и методы из области компьютерного зрения, обработки изображений и машинного обучения. Вот их подробное описание:

1. Цветовая фильтрация (Color Filtering)
   Цветовая фильтрация в коде используется для выделения определённых объектов на изображении, таких как дорожная разметка. Конкретно, применяются маски для выделения **белого** и **жёлтого** цветов с использованием пространства цветов **HSV** (Hue, Saturation, Value). Это позволяет эффективно выделить светлые или яркие участки изображения, что особенно полезно для обработки дорожных разметок и линий.

   - Преобразование BGR в HSV:
     - Базовый метод для выделения определённых цветов на изображении. Преобразование из пространства BGR в HSV делает фильтрацию цветов более инвариантной к изменениям освещённости, что улучшает стабильность работы алгоритма при различных условиях освещенности.

   - **Маски белого и жёлтого цветов**:
     - Белый цвет на изображениях характеризуется высокой яркостью (значения в V-канале) и низкой насыщенностью (в S-канале). Жёлтые линии, как правило, имеют высокую насыщенность и определённую яркость.
   
   - **Гауссово размытие (Gaussian Blur)**:
     - Используется для уменьшения шума и смягчения изображений, что улучшает качество последующих операций, таких как выделение контуров и обнаружение линий.

2. Выделение контуров (Edge Detection)
   После фильтрации изображения с помощью цветовых масок, алгоритм использует оператор **Кэнни** для выделения границ объектов в изображении. Этот метод является классическим и эффективным для нахождения резких изменений яркости, которые обычно соответствуют краям объектов, таких как линии разметки на дороге.

   - Оператор Кэнни:
     - Он включает несколько этапов: фильтрация изображения с помощью гауссова фильтра, вычисление градиента, подавление немаксимумов и использование пороговых значений для выделения краёв. Этот метод широко используется в обработке изображений для поиска границ объектов.

3. Прямые линии на изображении (Line Detection using Hough Transform)
   Для выявления прямых линий используется **преобразование Хафа** (Hough Transform). Этот метод позволяет находить прямые в изображениях, даже если они частично скрыты или прерываются.

   - Преобразование Хафа:
     - Преобразование Хафа является математическим методом для обнаружения геометрических фигур в изображениях, таких как линии, окружности и другие. В данном случае оно используется для нахождения линий на изображении, которые могут соответствовать разметке или границам движения.

   - Параметры для Хафа:
     - Параметры, такие как порог (threshold), минимальная длина линии и максимальный зазор между точками (maxLineGap), регулируют чувствительность метода к найденным линиям. Это позволяет гибко настроить алгоритм для различных условий.

4. Обработка трапециевидных областей интереса (ROI - Region of Interest)
   Алгоритм использует концепцию **области интереса** (ROI), которая представляет собой часть изображения, где происходит основная детекция. В коде используется трапециевидная область, определённая вручную, для выделения участков дороги, где происходит движение транспорта.

   - Трапециевидная форма ROI:
     - Использование трапециевидной формы полезно для отслеживания объектов, например, для выделения центральной полосы дороги или для выявления участков, где объекты могут пересекать линию. Такие области могут быть настроены для анализа только тех частей изображения, где происходят основные события.

5. Отслеживание объектов (Object Tracking)
   Детекции, сделанные на каждом кадре, сопоставляются друг с другом для отслеживания движения объектов. Для этого используется буфер, который хранит предыдущие детекции, а также метод для проверки **дубликатов нарушений**.

   - Буфер детекций (Detection Buffer):
     - С помощью буфера хранится история предыдущих детекций (например, координаты пересечений), чтобы исключить дублирующие записи в случае, если объект пересекает одну и ту же линию несколько раз в пределах заданного порога.

   - Гладкость (Smoothing) детекций:
     - Используется метод **скользящего среднего** для сглаживания координат детекций. Это позволяет уменьшить влияние случайных ошибок или шума в данных детекций, что особенно важно для динамичных сцен, где объекты могут двигаться с небольшой погрешностью.

6. **Нормализация координат (Normalization)
   Координаты детекций нормализуются относительно размеров кадра, преобразуя их в диапазон от 0 до 1. Это позволяет стандартизировать данные для дальнейшей обработки и анализа, а также уменьшить зависимость от размеров изображений.

7. Запись нарушений в CSV (Logging Violations in CSV)
   Все нарушения, такие как пересечение линий или нарушение стоп-линии, регистрируются в CSV файл. Для этого сохраняются:
   - Название видео.
   - Время события в формате **чч:мм:сс**.
   - Тип нарушения (например, пересечение линии или нарушение стоп-линии).

   CSV файл организуется с динамическим именованием, включающим название видео и текущую дату. Это облегчает идентификацию и организацию данных о нарушениях.

8. Подсчёт нарушений (Violation Counting)
   Счётчики нарушений отслеживают количество событий пересечения линии и нарушений стоп-линии. Это позволяет анализировать частоту нарушений в видеозаписи и использовать эти данные для дальнейшего анализа.

Заключение:
В коде реализованы основные принципы и методы из области компьютерного зрения, такие как фильтрация изображений, выделение контуров, преобразование Хафа для нахождения линий, а также методы нормализации и сглаживания данных. Все это позволяет эффективно обрабатывать видео для детекции нарушений и их логирования, что может быть использовано для анализа поведения транспортных средств в условиях реального времени.
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
