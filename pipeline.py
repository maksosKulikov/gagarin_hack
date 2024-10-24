import glob
import make_seg
import predict_class
import predict_text

names = glob.glob('data_base/valid/*/*.png')
predict_params = {"imgsz": 640, "conf": 0.1, "verbose": False, "device": "cpu", "max_det": 1}


classes = ['DRIVERS_SIDE_1',
           'DRIVERS_SIDE_2',
           'PASSPORT_SIDE_1',
           'PASSPORT_SIDE_2',
           'STS_SIDE_1',
           'STS_SIDE_2']

img_root_test = names[0]

make_seg.make_data(img_root_test)

label = predict_class.pred_label()
text = predict_text.predict_numbers(label)
print(classes[label], text)