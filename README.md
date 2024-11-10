# Transport_problems_hack

# ТИЗЕР
Для решения задачи обнаружения нарушений правил дорожного движения с использованием видео был применен следующий подход:

1. На первом этапе задается область интереса (ROI) в кадре, представляющая дорожное полотно. Кадр преобразуется в цветовое пространство HSV, чтобы выделить белые и желтые линии, и создается маска, на которой эти линии становятся более заметными.
2. Обработанная маска размытия и преобразуется с помощью оператора Кэнни для выделения краев. Затем применяется преобразование Хафа, чтобы находить линии на маске и классифицировать их по углу наклона.
3. Если найденные линии проходят через заданные области интереса, алгоритм фиксирует пересечение или нарушение стоп-линии, увеличивая соответствующий счетчик.
4. Далее идёт определение дорожных знаков с помощью до обученной модели YOLO. 

**Технические особенности:** 

Python, OpenCV, Numpy, фильтры HSV, метод Кэнни для выделения краев, преобразование Хафа, YOLO.

**Уникальность:**

Алгоритм настроен на определение двух ключевых нарушений — пересечение полосы и нарушение стоп-линии.


# Описание работы детектора:

В коде применяются различные научные принципы и методы из области компьютерного зрения, обработки изображений и машинного обучения. Вот их подробное описание:

### 1. Цветовая фильтрация (Color Filtering)
Цветовая фильтрация используется для выделения определённых объектов на изображении, таких как дорожная разметка. Конкретно, применяются маски для выделения **белого** и **жёлтого** цветов с использованием пространства цветов **HSV** (Hue, Saturation, Value). Это позволяет эффективно выделить светлые или яркие участки изображения, что особенно полезно для обработки дорожных разметок и линий.

- **Преобразование BGR в HSV**:
    - Базовый метод для выделения определённых цветов на изображении. Преобразование из пространства BGR в HSV делает фильтрацию цветов более инвариантной к изменениям освещённости, что улучшает стабильность работы алгоритма при различных условиях освещенности.
  
- **Маски белого и жёлтого цветов**:
    - Белый цвет на изображениях характеризуется высокой яркостью (значения в V-канале) и низкой насыщенностью (в S-канале). Жёлтые линии, как правило, имеют высокую насыщенность и определённую яркость.
  
- **Гауссово размытие (Gaussian Blur)**:
    - Используется для уменьшения шума и смягчения изображений, что улучшает качество последующих операций, таких как выделение контуров и обнаружение линий.

### 2. Выделение контуров (Edge Detection)
После фильтрации изображения с помощью цветовых масок, алгоритм использует оператор **Кэнни** для выделения границ объектов в изображении. Этот метод является классическим и эффективным для нахождения резких изменений яркости, которые обычно соответствуют краям объектов, таких как линии разметки на дороге.

- **Оператор Кэнни**:
    - Он включает несколько этапов: фильтрация изображения с помощью гауссова фильтра, вычисление градиента, подавление немаксимумов и использование пороговых значений для выделения краёв. Этот метод широко используется в обработке изображений для поиска границ объектов.

### 3. Прямые линии на изображении (Line Detection using Hough Transform)
Для выявления прямых линий используется **преобразование Хафа** (Hough Transform). Этот метод позволяет находить прямые в изображениях, даже если они частично скрыты или прерываются.

- **Преобразование Хафа**:
    - Преобразование Хафа является математическим методом для обнаружения геометрических фигур в изображениях, таких как линии, окружности и другие. В данном случае оно используется для нахождения линий на изображении, которые могут соответствовать разметке или границам движения.

- **Параметры для Хафа**:
    - Параметры, такие как порог (threshold), минимальная длина линии и максимальный зазор между точками (maxLineGap), регулируют чувствительность метода к найденным линиям. Это позволяет гибко настроить алгоритм для различных условий.

### 4. Обработка трапециевидных областей интереса (ROI - Region of Interest)
Алгоритм использует концепцию **области интереса** (ROI), которая представляет собой часть изображения, где происходит основная детекция. В коде используется трапециевидная область, определённая вручную, для выделения участков дороги, где происходит движение транспорта.

- **Трапециевидная форма ROI**:
    - Использование трапециевидной формы полезно для отслеживания объектов, например, для выделения центральной полосы дороги или для выявления участков, где объекты могут пересекать линию. Такие области могут быть настроены для анализа только тех частей изображения, где происходят основные события.

### 5. Отслеживание объектов (Object Tracking)
Детекции, сделанные на каждом кадре, сопоставляются друг с другом для отслеживания движения объектов. Для этого используется буфер, который хранит предыдущие детекции, а также метод для проверки **дубликатов нарушений**.

- **Буфер детекций (Detection Buffer)**:
    - С помощью буфера хранится история предыдущих детекций (например, координаты пересечений), чтобы исключить дублирующие записи в случае, если объект пересекает одну и ту же линию несколько раз в пределах заданного порога.

- **Гладкость (Smoothing) детекций**:
    - Используется метод **скользящего среднего** для сглаживания координат детекций. Это позволяет уменьшить влияние случайных ошибок или шума в данных детекций, что особенно важно для динамичных сцен, где объекты могут двигаться с небольшой погрешностью.

### 6. Нормализация координат (Normalization)
Координаты детекций нормализуются относительно размеров кадра, преобразуя их в диапазон от 0 до 1. Это позволяет стандартизировать данные для дальнейшей обработки и анализа, а также уменьшить зависимость от размеров изображений.

### 7. Запись нарушений в CSV (Logging Violations in CSV)
Все нарушения, такие как пересечение линий или нарушение стоп-линии, регистрируются в CSV файл. Для этого сохраняются:
- Название видео.
- Время события в формате **чч:мм:сс**.
- Тип нарушения (например, пересечение линии или нарушение стоп-линии).

CSV файл организуется с динамическим именованием, включающим название видео и текущую дату. Это облегчает идентификацию и организацию данных о нарушениях.

### 8. Подсчёт нарушений (Violation Counting)
Счётчики нарушений отслеживают количество событий пересечения линии и нарушений стоп-линии. Это позволяет анализировать частоту нарушений в видеозаписи и использовать эти данные для дальнейшего анализа.
