from ultralytics import YOLO
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO("yolo11m-seg.pt")

if __name__ == '__main__':
    model.train(data="C:/Users/96224/PycharmProjects/gagarin_hack/datasets/data_new/data.yaml", epochs=100, device=0,
                imgsz=640, batch=3, mosaic=0)
