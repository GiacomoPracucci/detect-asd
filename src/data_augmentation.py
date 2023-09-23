from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import glob
import os
import shutil
import cv2
import os

# from the original folders "TCImages" and "TSImages" we create "TCaugmeted" and "TSaugmented" folders
# each original image will be "augmented" by creating 5 more (imgCount) modified copies

class DataAugmentation:
    def __init__(self, imgCount=5): #Setting the initial parameters for the image augmentation process
        self.augmenter = ImageDataGenerator(
            rotation_range=10,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest')
        self.imgCount = imgCount

    def augment_images(self, input_directory, output_directory, prefix):
        print(f"Augmenting Images in {input_directory}:")
        
        os.makedirs(output_directory, exist_ok=True)

        for file in glob.iglob(f'{input_directory}/*.png'):
            print("Current File:", file)
            
            try:
                img = load_img(file) 
                img = img_to_array(img)  # Numpy array with shape (3, 150, 150)
                img = img.reshape((1,) + img.shape)  # Numpy array with shape (1, 3, 150, 150)
            except Exception as e:
                print(f"Error loading image {file}: {e}")
                continue

            # also save original image in output directory
            shutil.copy(file, output_directory)

            i = 0
            for batch in self.augmenter.flow(img, batch_size=1,
                                    save_to_dir=output_directory, 
                                    save_prefix=f'{prefix}_Aug_{os.path.basename(file).split(".")[0]}', 
                                    save_format='png'):
                i += 1
                if i == self.imgCount:
                    break

# this class loads images, applies a threshold to get a binary image, 
# finds the outline with the maximum area, crops the original image based on the outline, and resizes 
# the cropped image to the desired size. Finally, save the cropped images in two different directories 
# based on their content (negative or positive).
class ImageCropper:
    def __init__(self, imgDim=224):
        self.imgDim = imgDim

    def process_and_save(self, input_directory, output_directory):
        for file in glob.iglob(f'{input_directory}/*.png'):
            print("Current File:", file)
            img = cv2.imread(file)
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Thresholding
            th, threshed = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY)
            
            # Find the max-area contour
            if cv2.__version__.startswith('3.'):
                _, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cnt = sorted(contours, key=cv2.contourArea)[-1]
            x, y, w, h = cv2.boundingRect(cnt)
            imgCropped = img[y:y + h, x:x + w]  # Cropped image

            imgCropped = cv2.resize(imgCropped, (self.imgDim, self.imgDim))  # resize
            
            # Saving cropped image
            baseName = os.path.basename(file)
            fileName, fileExtension = os.path.splitext(baseName)
            cv2.imwrite(f'{output_directory}/{fileName}.png', imgCropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == "__main__":
    # Data Augmentation
    augmenter = DataAugmentation(imgCount=5)
    augmenter.augment_images('TCImages', 'TCaugmented', 'TC')
    augmenter.augment_images('TSImages', 'TSaugmented', 'TS')
    
    # Image Cropping
    cropper = ImageCropper(imgDim=224)
    cropper.process_and_save('TCAugmented', 'TCcrop')
    cropper.process_and_save('TSaugmented', 'TScrop')