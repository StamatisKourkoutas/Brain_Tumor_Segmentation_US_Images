"""
Code taken from:
    https://github.com/utkuozbulak/pytorch-cnn-visualizations
and is altered as required.
"""
import torch
from torch.nn import ReLU
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back-propagation
       or vanila back-propagation from the given image
    """
    def __init__(self, model, mode="vanila"):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        if (mode=="guided"):
            self.update_relus()
        self.hook_layers()


    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list((list(self.model.children())[0]).children())[0]
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
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)


    def generate_gradients(self, input_image, target_mask):
        # Forward pass
        model_atts, model_dets = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Backward pass
        model_dets.backward(target_mask)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]

        return gradients_as_arr

if __name__ == '__main__':
    (original_image, prep_img, target_mask, file_name_to_export, pretrained_model) = get_params()

    # Guided back-propagation
    GBP = GuidedBackprop(pretrained_model, "guided")
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
    print('Back-propagation completed')
