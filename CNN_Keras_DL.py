
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 128, 128



train_data_dir ='/home/ubuntu/data/Train'

validation_data_dir = '/home/ubuntu/data/Validation'

test_data_dir = '/home/ubuntu/data/Test'

nb_train_samples = 7277

nb_validation_samples = 1301

epochs = 40

batch_size = 32





input_shape = (img_width, img_height, 1)




model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape, dim_ordering='tf'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(3))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

             optimizer='rmsprop',

             metrics=['accuracy'])



# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(rescale=1. / 255)



# this is the augmentation configuration we will use for testing:

# only rescaling

val_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(
      train_data_dir,

      target_size=(img_width, img_height),

      batch_size=batch_size,
    
      color_mode='grayscale',

      class_mode='categorical')



validation_generator = val_datagen.flow_from_directory(

     validation_data_dir,

     target_size=(img_width, img_height),

     batch_size=batch_size,
     
     color_mode='grayscale',

     class_mode='categorical')


test_generator = test_datagen.flow_from_directory(

     test_data_dir,

     target_size=(img_width, img_height),

     batch_size=batch_size,
     
     color_mode='grayscale',

     class_mode='categorical')


model.fit_generator(

      train_generator,

      steps_per_epoch=nb_train_samples // batch_size,

      epochs=epochs,

      validation_data=validation_generator,

      validation_steps=nb_validation_samples // batch_size)


score = model.evaluate_generator(test_generator, 936)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights('/home/ubuntu/first_try.h5')