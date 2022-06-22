import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def plot_real_images(images, device):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Real images")
    plt.imshow(
        np.transpose(vutils.make_grid(images[0].to(device)[:64], padding=2, normalize=True).cpu(),
                     (1, 2, 0)))
    plt.show()


def plot_fake_images(images):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(images[-1], (1, 2, 0)))
    plt.show()


def plot_images_with_labels(images, real_labels, predicted_labels=False, device='cpu'):
    plt.figure(figsize=(8, 8))
    fig, axs = plt.subplots(1, 4)

    for i in range(4):
        axs[i].imshow(
            np.transpose(vutils.make_grid(images.to(device)[i], padding=100, normalize=True).cpu(),
                         (1, 2, 0)))

        if predicted_labels:
            image_title = f'P: {predicted_labels[i]}\nT: {real_labels[i]}'
        else:
            image_title = f'T: {real_labels[i]}'

        axs[i].set_title(image_title)
        axs[i].axis('off')
