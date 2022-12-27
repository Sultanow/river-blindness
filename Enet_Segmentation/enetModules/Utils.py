import torchvision.transforms.functional as T
import numpy as np

class Utils:
    def imageAugmentation(image):
        '''
        This function augments the given image by rotating it.
        
        Arguments:
        - image : Image to augment

        Return:
        - augmented images : Returns an array of augmented images including the original image
        '''
        augmented_images = []
       
        augmented_images.append(image)
        augmented_images.append(T.rotate(image, 45, fill=0))
        augmented_images.append(T.rotate(image, 135, fill=0))    

        return augmented_images

    def get_class_weights(labels, num_classes, c=1.02):
        '''
        This class return the class weights for each class
        
        Arguments:
        - labels : All labels in this trainigs session

        - num_classes : The number of classes

        Return:
        - class_weights : An array equal in length to the number of classes
                        containing the class weights for each class
        '''

        all_labels = labels.flatten()
        each_class = np.bincount(all_labels, minlength=num_classes)
        prospensity_score = each_class / len(all_labels)
        class_weights = 1 / (np.log(c + prospensity_score))
        return class_weights

    def decode_segmap(image, color_map):
        color_values = []
        
        for color in color_map.values():
            color_values.append(color)
            
        label_colors = np.array(color_values).astype(np.uint8)
        
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for label in range(len(label_colors)):
            r[image == label] = label_colors[label, 0]
            g[image == label] = label_colors[label, 1]
            b[image == label] = label_colors[label, 2]
            
        rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b        
        
        return rgb