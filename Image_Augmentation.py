from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import PIL
import glob

for filename in glob.glob("/home/rambo/Images/Stone/*.JPG"):
    img = load_img(filename)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/rambo/Images/Kstone', save_prefix='stone', save_format='jpeg'):
        i += 1
        if i > 5:
            break 