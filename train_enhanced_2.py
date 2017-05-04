import os
import glob
import pre_process
import cnn_model_batch_norm
import numpy as np
from skimage import io
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K

def get_class(img_path):
    return int(img_path.split('/')[-2])

K.set_image_dim_ordering('th')
root_dir = 'GTSRB/Final_Training/Images/'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img = pre_process.preprocess_img(io.imread(img_path))
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(pre_process.NUM_CLASSES, dtype='uint8')[labels]

model = cnn_model_batch_norm.cnn_model()

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
                    ModelCheckpoint('enhance_2_model.h5',save_best_only=True)]
         )
