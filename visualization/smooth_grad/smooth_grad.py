"""
Code taken from:
    https://github.com/utkuozbulak/pytorch-cnn-visualizations
and is altered as required.
"""
import numpy as np
from torch.autograd import Variable
import torch

import sys
sys.path.insert(1, '../guided_backpropagation')
from guided_backprop import GuidedBackprop
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images)


def generate_smooth_grad(Backprop, prep_img, target_mask, param_n, param_sigma_multiplier):
    """
        Generates smooth gradients of given Backprop type. You can use this with both vanilla
        and guided backprop
    Args:
        prep_img (torch Variable): preprocessed image
        target_class (int): target class
        param_n (int): Amount of images used to smooth gradient
        param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
    """
    # Generate an empty image/matrix
    smooth_grad = np.zeros(prep_img.size()[1:])

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
    for x in range(param_n):
        # Generate noise
        noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma**2))
        # Add noise to the image
        noisy_img = prep_img + noise
        # Calculate gradients
        grads = Backprop.generate_gradients(noisy_img, target_mask)
        # Add gradients to smooth_grad
        smooth_grad = smooth_grad + grads
    # Average it out
    smooth_grad = smooth_grad / param_n
    return smooth_grad


if __name__ == '__main__':
    (original_image, prep_img, target_mask, file_name_to_export, pretrained_model) = get_params()

    # Vanila back-propagation
    GBP = GuidedBackprop(pretrained_model, "vanila")

    param_n = 50
    param_sigma_multiplier = 2
    smooth_grad = generate_smooth_grad(GBP,
                                       prep_img,
                                       target_mask,
                                       param_n,
                                       param_sigma_multiplier)

    # Save colored gradients
    save_gradient_images(smooth_grad, file_name_to_export + '_SmoothGrad_color')
    # Convert to grayscale
    grayscale_smooth_grad = convert_to_grayscale(smooth_grad)
    # Save grayscale gradients
    save_gradient_images(grayscale_smooth_grad, file_name_to_export + '_SmoothGrad_gray')
    print('Smooth grad completed')
