import os
import glob
import pre_process_last
import vgg16_model
import numpy as np
from skimage import io
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential

def get_class(img_path):
    return int(img_path.split('/')[-2])

root_dir = 'GTSRB/Final_Training/Images/'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img = pre_process_last.preprocess_img(io.imread(img_path))
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(pre_process_last.NUM_CLASSES, dtype='uint8')[labels]

model = vgg16_model.vgg16_model((pre_process_last.IMG_SIZE, pre_process_last.IMG_SIZE, 3), pre_process_last.NUM_CLASSES)

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(pre_process_last.NUM_CLASSES, activation='softmax'))

model.add(top_model)

# let's train the model using SGD + momentum
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])
def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

batch_size = 32
nb_epoch = 30

model.fit(X, Y,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                    ModelCheckpoint('enhance_1_model.h5',save_best_only=True)]
         )
