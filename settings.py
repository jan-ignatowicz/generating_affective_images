"""Settings and HyperParameters used in the project"""

import os

####################################################################################################
# SETTINGS
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

# Root directory for augmented dataset
all_images_path = f"{ROOT_DIR}/datasets/all_data_affective/images/"
annonations_file = f"{ROOT_DIR}/datasets/TrainingData.csv"

augmented_images_path = f"{ROOT_DIR}/datasets/all_data_affective/augmented/"
augmented_annonations_file = f"{ROOT_DIR}/datasets/AugmentedTrainingData.csv"

# path to netG (to continue training)
netG = ""

# path to netD (to continue training)
netD = ""

# folder to output images
outi = "./generated/"

# folder to output model checkpoints
outc = "./checkpoints/"

###################################################################################################
# HyperParameters

# Number of training epochs
epochs = 100

# Batch size of counting FID and KID
kid_batch = fid_batch = 200

# define interval between counting FID and KID
kid_interval = fid_interval = 5000

log_interval = 400

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
num_channels = 3

# Size of z latent vector (i.e. size of generator input)
latent_dim = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers (different for both network; according to TTUR)
# manipulation of lrs is better than learning one network for i.e. 3 epochs and the
# other for 1 epoch
lr_D = 0.0003
lr_G = 0.0001

# Beta1 hyperparam for Adam optimizer (for generator)
beta1 = 0.5
beta2 = 0.999

####################################################################################################
# ACGAN HyperParameters

# number of classes for data
n_classes = 13

labels_map = {0: "neutral", 1: "content", 2: "relaxed", 3: "calm", 4: "tired", 5: "bored",
              6: "depressed", 7: "frustrated", 8: "angry", 9: "tense", 10: "excited",
              11: "delighted", 12: "happy"}
