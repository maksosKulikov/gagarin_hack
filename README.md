#### Задача хакатона:
Определить тип и страницу документа, вернуть сирию и номер документа. Если модель получила объект в повёрнутом/перевёрнутом виде, то модель его должна перевернуть.
#### Пример работы:
https://github.com/user-attachments/assets/4e496aa0-c966-4ab4-9477-e63ea9515fc0
#### Датасет:
Дано 3 типа документов, для каждого по 2 страницы/оборота (итого - 6 классов). В общей сложности было дано около 200 изображений.
#### Решение:
Были размечены документы для задачи сегментации. После этого была обучена модель yolo11 для задачи сегментации. Я после работы модели нахожу 4 крайние точки обнаруженного объекта и с помощью перспективной трансформации (cv2.warpPerspective()) получаю изображение в виде прямоугольника. Далее с помощью ResNet определяю класс документа. После этого разметил датасет для задачи object detection, обучаю другую модель yolo11, которая позволяет мне определить прямоугольник, в который заключены серия и номер документов. С помощью предобученной модели tesseract получаю сирию и номер документа. От того как раположен прямоугольник с серией и номером и от того насколько правильно tesseract распознаёт цифры я понимаю как мне нужно перевернуть изображение, полученное после перспективной трансформации (логика переворачивания изображения написана в predict_text.py). Всю работу запаковал в telegram bot
#### Датасет и модели:
https://disk.yandex.ru/d/D-Vp-8ZfR6qgCg
