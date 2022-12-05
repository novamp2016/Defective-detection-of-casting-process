# 라이브러리 호출
import os, cv2, csv, sqlite3, time
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
from PIL import Image


# 패키지 호출
from yolo3.yolo import YOLO, detect_video

abs_path = 'C:/Users/novam/project4/checkpoints/'

casting_yolo = YOLO(model_path= abs_path + 'trained_weights_casting.h5',
                    anchors_path= abs_path + 'yolo_anchors.txt',
                    classes_path= abs_path + 'casting_classes.txt',
                    score=0.55)

helmet_yolo = YOLO(model_path= abs_path + 'trained_weights_helmet.h5',
                    anchors_path= abs_path + 'yolo_anchors.txt',
                    classes_path= abs_path + 'helmet_classes.txt',
                    score=0.4)


def test_image(x):
    img_1 = os.listdir('./data/casting_1/test/def_front/')[x]
    load_image = Image.open(f'./data/casting_1/test/def_front/{img_1}')
    resize_image = load_image.resize((416,416))
    final_img = casting_yolo.detect_image(resize_image)[0]
    plt.imshow(final_img)


def detect_images(folder_name='casting_1', base='casting', target='hole'):

    image_list = os.listdir(f'./data/{folder_name}/detect/')
    hole_list = []

    for image in image_list:
        load_image = Image.open(f'./data/{folder_name}/detect/{image}')
        resize_image = load_image.resize((416,416))
        detected_img, label_list, coordinate_list = casting_yolo.detect_image(resize_image)
        label_num = label_list.count(target)

        if label_num > 0: 
            base_co = coordinate_list.pop(label_list.index(base))
            top, bottom, left ,right = base_co
            R = (bottom-top+right-left)/4 # casting 바운딩 박스의 x,y 길이 평균의 반을 원의 반지름으로 설정
            
            for coordinate in coordinate_list:
                a,b,c,d = coordinate
                temp = [[a,c],[a,d],[b,c],[b,d]]
                trigger = False

                for y,x in temp: # casting의 원밖으로 나가는 hole 바운딩 박스가 나올시 pin_hole 불량이 아님으로 판정
                    if ((x-(left+(right-left)/2))**2)+((y-(top+(bottom-top)/2))**2) > R**2:
                        os.makedirs(f'./data/{folder_name}/detected/target_contour/', exist_ok=True)
                        detected_img.save(f'./data/{folder_name}/detected/target_contour/{image}','JPEG')
                        trigger = True
                        break

                if trigger == True: break
                else: hole_list.append([c+(d-c)/2, a+(b-a)/2, (b-a)*(d-c)]) # [중앙 좌표, 크기]

            if trigger == True: continue

        os.makedirs(f'./data/{folder_name}/detected/target_{label_num}/', exist_ok=True)
        detected_img.save(f'./data/{folder_name}/detected/target_{label_num}/{image}','JPEG')
    
    f = open(f'./data/{folder_name}/detected/hole.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for hole in hole_list: # hole 좌표, 크기 csv 파일로 저장
        wr.writerow(hole)
    f.close()


def scatter_plot(folder_name='casting_1'): # scatter_plot('casting_1')
    f = open(f'./data/{folder_name}/hole.csv', 'r', encoding='utf-8', newline='')
    rdr = csv.reader(f)
    x_list = []
    y_list = []
    size_list = []
    for x, y, size in rdr:
        x_list.append(float(x))
        y_list.append(float(y))
        size_list.append(float(size))
    f.close()

    plt.scatter(x=x_list[:100], y=y_list[:100], c=size_list[:100], cmap='Greens')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.title('Hole Scatter')
    plt.colorbar(label='size')
    plt.savefig('./data/plt_image/scatter_{}.png'.format(folder_name))


def count_detect_image(folder_name):
    count_dic = {}
    folder_list = os.listdir(f'./data/{folder_name}/detected/')
    for folder in folder_list:
        image_list = os.listdir(f'./data/{folder_name}/detected/{folder}')
        count_dic[folder] = len(image_list)
    return count_dic


def test_video():
    detect_video(helmet_yolo,'./data/videos/test_video.mov', './data/videos/detect_video.mov')

            
def train_test(folder_name): # train, test 셋 설정
    train = tf.keras.preprocessing.image_dataset_from_directory(
        './data/{}/train'.format(folder_name),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(300, 300))

    test = tf.keras.preprocessing.image_dataset_from_directory(
        './data/{}/test'.format(folder_name),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(300, 300))
    
    return train, test


def scan_dir(dir_name=''): # 폴더 속 파일 리스트 추출
    dir_list = os.listdir('./data/{}'.format(dir_name))
    return dir_list


def training(model_list, batch_size=30, epochs=20, patience=8, dir_list=scan_dir()):
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                                   min_delta = 0,
                                   patience = patience,
                                   mode = 'max')

    for dir_name in dir_list:
        train, test = train_test(dir_name)
        for deep_model in model_list:

            checkpoint_filepath = './checkpoints/{}.h5'.format(deep_model)

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

            tuner = kt.Hyperband(hypermodel = globals()['{}'.format(deep_model)],
                     objective = 'val_accuracy', 
                     max_epochs = epochs,
                     factor = 3,
                     directory = 'checkpoints',
                     project_name = deep_model)
            
            tuner.search(train, validation_data=test)
            
            best_hps = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hps)

            hist = model.fit(train, 
                  batch_size=batch_size,
                  validation_data=test,
                  epochs=100,
                  callbacks=[early_stopping, model_checkpoint_callback])

            save_plt(hist, dir_name)
            print('{} 모델 저장 완료.'.format(dir_name))


def load_model(model_name):
    checkpoint_filepath = './checkpoints/{}.h5'.format(model_name)
    model = keras.models.load_model(checkpoint_filepath)
    return model


def evalate_model(dir_name, model_name):
    test = train_test(dir_name)[1]
    model = load_model(model_name)
    evalate = model.evaluate(test)
    return evalate


def CNN(hp):
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
    
    hp_unit1 = hp.Int('units_1', min_value = 24, max_value = 48, step = 4)

    inputs = tf.keras.Input(shape=[300, 300, 3])
    x = inputs

    x = Conv2D(hp_unit1, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(2*hp_unit1, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(2*hp_unit1, 3, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(hp.Int('dense_num', min_value = 64, max_value = 128, step=16), activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
    
    return model


def save_plt(hist, dir_name): # 학습 그래프 저장
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label = 'train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label = 'val loss')
    acc_ax.plot(hist.history['accuracy'], 'b', label = 'train accuracy')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label = 'val accuracy')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_xlabel('accuracy')

    loss_ax.legend(loc = 'upper left')
    acc_ax.legend(loc = 'lower left')
    plt.savefig('.data/plt_image/model_{}.png'.format(dir_name))


def start_sql(): # sqlite db 파일 생성 및 연결 후 테이블 생성
    global conn, cur
    DB_FILENAME = 'log_DB.db' 
    DB_FILEPATH = os.path.join(os.getcwd(), DB_FILENAME)
    conn = sqlite3.connect(DB_FILEPATH)
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM sqlite_master where name = "log"')
    result_1 = cur.fetchone()[0]
    
    if result_1 == 0:
        cur.execute("""
        CREATE TABLE log
        (
        Id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP,
        label INTEGER(1),
        predict INTEGER(1),
        reason VARCHAR
        );
			""")
        conn.commit()


def product_log(predict, reason=None): # timestamp, predict, reason 입력 (모델 분류 모델과 연결)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    cur.execute('INSERT INTO log (timestamp, predict, reason) VALUES ("{}",{},"{}");'.format(now, predict, reason))
    conn.commit()

