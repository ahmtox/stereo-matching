import numpy as np
import matplotlib.pyplot as plt


class StereogramGenerator:

    def __init__(self):
        self.binary_image = None
        self.shifted_image = None

    def generate_stereogram(self, height, width, horizontal_shift):

        image = np.random.rand(height, width)
        binary_image = (image > 0.5)*1.0

        height_range = [int(height/4), int(height*3/4)]
        width_range = [int(width/4), int(width*3/4)]

        shifted_area = binary_image[height_range[0]: height_range[1], width_range[0]: width_range[1]]
        shifted_image = binary_image.copy()
        shifted_image[height_range[0]: height_range[1], width_range[0] - horizontal_shift: width_range[1] - horizontal_shift] = shifted_area

        self.binary_image = binary_image
        self.shifted_image = shifted_image

        return binary_image, shifted_image
    
    def plot_stereogram(self):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        ax[0].imshow(self.binary_image, cmap='gray')
        ax[0].set_title('Image 1')
        ax[0].axis('off')

        ax[1].imshow(self.shifted_image, cmap='gray')
        ax[1].set_title('Image 2')
        ax[1].axis('off')

        ax[2].imshow(self.shifted_image - self.binary_image, cmap='gray')
        ax[2].set_title('difference')
        ax[2].axis('off')

        plt.tight_layout()
        plt.show()

    def compute_disparity_map(self):
        height, width = self.binary_image.shape
        disparity_map = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                found_match = False
                for k in range(-10, 11):
                    if j + k >= 0 and j + k < width:
                        if self.binary_image[i, j] == self.shifted_image[i, j + k]:
                            disparity_map[i, j] = k 
                            found_match = True
                            break

                if not found_match:
                    disparity_map[i, j] = np.nan  

        return disparity_map

    def plot_disparity_map(self, disparity_map):
        plt.figure(figsize=(10, 6))
        
        plt.imshow(disparity_map, cmap='coolwarm', aspect='auto', interpolation='nearest')

        plt.colorbar(label='Disparity')

        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title('Disparity Map')

        plt.show()

