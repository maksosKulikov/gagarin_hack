import telebot
import glob
import make_seg
import predict_class
import predict_text

bot = telebot.TeleBot('7995645909:AAGrXraTklYFQ5mPzBpWjsHazWSA736do2A')

names = glob.glob('data_base/valid/*/*.png')
predict_params = {"imgsz": 640, "conf": 0.1, "verbose": False, "device": "cpu", "max_det": 1}

classes = ['DRIVERS_SIDE_1',
           'DRIVERS_SIDE_2',
           'PASSPORT_SIDE_1',
           'PASSPORT_SIDE_2',
           'STS_SIDE_1',
           'STS_SIDE_2']


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Отправьте фото документа")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    save_path = 'need_predict/base_img.png'
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.reply_to(message, 'Информация обрабатывается...')

    seg_done = make_seg.make_data(save_path)
    label = predict_class.predict_label()
    text = predict_text.predict_numbers(label)
    print(classes[label], text)

    chat_id = message.chat.id

    if not seg_done:
        bot.send_message(chat_id, f"На изображении не распознан документ :(")
    elif text == False:
        bot.send_message(chat_id, f"Информацию не получилось распознать :(")
    elif label == 3:
        bot.send_message(chat_id, f"Класс документа: {classes[label]}")
        bot.send_message(chat_id, text)
    else:
        bot.send_photo(chat_id, photo=open('need_predict/X/image_after_segment.png', 'rb'))
        bot.send_message(chat_id, f"Класс документа: {classes[label]}")
        bot.send_message(chat_id, f"Серия: {text[:4]} и номер: {text[4:]}")


bot.infinity_polling()
