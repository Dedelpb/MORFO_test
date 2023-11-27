import numpy as np
import pandas
import pyarrow
import matplotlib.pyplot as plt

#Question 1 
def random_rgb_image(shape=(256, 512, 3)):
    return np.random.randint(0, 256, shape, dtype=np.uint8)

num_batches = 5
images_per_batch = 5
image_batches = []

for batch in range(num_batches):
    image_batch = []
    for image_index in range(images_per_batch):
        img = random_rgb_image()
        image_batch.append(img)
    image_batches.append(image_batch)
    
test_image = image_batches[2][2]

#Test if ok
# print(test_image.shape)
# print(test_image)

#Question 2
def image_with_squares(image,shape=(256, 512, 3), square_size=100):
    
    #Coords of the top left of the 1st rectangle
    top_left_1_1 = np.random.randint(0, shape[0] - square_size + 1)
    top_left_1_2 = np.random.randint(0, shape[1] - square_size + 1)

    #Coords of the top left of the 2nd rectangle
    top_left_2_1 = np.random.randint(0, shape[0] - square_size + 1)
    top_left_2_2 = np.random.randint(0, shape[1] - square_size + 1)

    #Test no overlap
    while np.abs(top_left_2_1 - top_left_1_1 ) and np.abs(top_left_2_2 - top_left_1_2) < square_size:
        top_left_2_1 = np.random.randint(0, shape[0] - square_size + 1, 1)
        top_left_2_2 = np.random.randint(0, shape[1] - square_size + 1, 1)

    #1st rectangle
    image[top_left_1_1:top_left_1_1 + square_size, top_left_1_2:top_left_1_2 + square_size, :] = 255

    #2nd rectangle
    image[top_left_2_1:top_left_2_1 + square_size, top_left_2_2:top_left_2_2 + square_size, :] = 0

    return image

test_image_square = image_with_squares(test_image)

plt.imshow(test_image_square)
plt.show()

#Question 3
def random_crop(image, crop_size=200):
    h, w, _ = image.shape

    #generate 2D coords for the crop
    top_left_1 = np.random.randint(0, h - crop_size + 1)
    top_left_2 = np.random.randint(0, w - crop_size + 1)

    #Crop
    cropped_image = image[top_left_1:top_left_1 + crop_size, top_left_2:top_left_2 + crop_size, :]

    return cropped_image

crop_image = random_crop(test_image)

plt.imshow(crop_image)
plt.show()

#Question 4
def calculate_avg_std(image_batch):
    white_pixels = []
    black_pixels = []

    for image in image_batch:
        white_pixels.append(np.sum(image == 255))
        black_pixels.append(np.sum(image == 0))
        
    #Average and std
    avg_white_pixels = np.mean(white_pixels)
    std_white_pixels = np.std(white_pixels)
    avg_black_pixels = np.mean(black_pixels)
    std_black_pixels = np.std(black_pixels)

    return avg_white_pixels, std_white_pixels, avg_black_pixels, std_black_pixels
    




