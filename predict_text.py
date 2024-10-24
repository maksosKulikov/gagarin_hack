import pytesseract
from ultralytics import YOLO
import cv2 as cv
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('train_models/runs/detect/train/weights/best.pt')
predict_params = {"imgsz": 640, "conf": 0.1, "verbose": False, "device": "cpu", "max_det": 1}

img_root = "need_predict/X/image_after_segment.png"


def predict_numbers(label):
    if label == 3:
        return "Для данного типа документа мы не умеем распознавать информацию"

    results = model(img_root, **predict_params)
    try:
        x1, y1, x2, y2 = results[0].boxes.xyxy[0]
    except:
        return False
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    img = cv.imread(img_root)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img[y1:y2, x1:x2]

    if y2 - y1 > x2 - x1:
        if label == 2:
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

            img_document = cv.imread(img_root)
            img_document = cv.rotate(img_document, cv.ROTATE_90_CLOCKWISE)
            cv.imwrite(img_root, img_document)
    elif label == 2:
        img_document = cv.imread(img_root)
        img_document = cv.rotate(img_document, cv.ROTATE_90_CLOCKWISE)
        cv.imwrite(img_root, img_document)

    img = cv.resize(img, (300, 100))
    cv.imwrite("need_predict/X/text.jpg", img)

    img = cv.imread("need_predict/X/text.jpg", 0)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    img = cv.GaussianBlur(img, (3, 3), 0)
    text = pytesseract.image_to_string(img, lang='eng',
                                       config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    print(text)
    img = cv.imread("need_predict/X/text.jpg")

    need_predict = False

    if len(text) < 8:
        img = cv.rotate(img, cv.ROTATE_180)

        img_document = cv.imread(img_root)
        img_document = cv.rotate(img_document, cv.ROTATE_180)
        cv.imwrite(img_root, img_document)

        need_predict = True

    cv.imwrite("need_predict/X/text.jpg", img)

    if need_predict:
        img = cv.imread("need_predict/X/text.jpg", 0)
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        img = cv.GaussianBlur(img, (3, 3), 0)
        text = pytesseract.image_to_string(img, lang='eng',
                                           config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

    return text
