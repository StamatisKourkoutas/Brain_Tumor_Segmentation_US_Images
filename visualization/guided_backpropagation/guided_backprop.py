"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU

from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
CE = torch.nn.BCEWithLogitsLoss()


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        #self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()
        #self.xx = 0

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]
        #first_layer = list(self.model.children())[0] 				#For ResNet
        first_layer = list((list(self.model.children())[0]).children())[0]	#For CPD-R
        first_layer.register_backward_hook(hook_function)


    def update_relus(self):
        """
        Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                #self.xx=self.xx+1
                #print(self.xx)
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)


    def generate_gradients(self, input_image, target_mask):
        # Forward pass
        model_atts, model_dets = self.model(input_image)
        #print(model_atts.size())
        
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, 1, 224, 224).zero_()
        one_hot_output = target_mask
        model_output = model_dets + model_atts
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]

        return gradients_as_arr


if __name__ == '__main__':
    (original_image, prep_img, target_mask, file_name_to_export, pretrained_model) = get_params()

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_mask)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')
