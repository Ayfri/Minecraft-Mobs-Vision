"""Central training / preprocessing configuration.

All tuneable knobs live here so experiments can be compared by changing a single line.
Scripts import from this module - never hard-code these values elsewhere.

Backbone options (timm model names, all accept pretrained=True):
  - "mobilenetv4_conv_small.e2400_r224_in1k"   : 2.5 M params,  fastest  ← default
  - "mobilenetv4_conv_medium.e500_r224_in1k"   : 9.7 M params,  balanced
  - "efficientnet_b0.ra_in1k"                  : 5.3 M params,  previous default
"""

BACKBONE: str = "mobilenetv4_conv_small.e2400_r224_in1k"

# File extension of the captured frames ("jpg" or "png").
# Must match the imageFormat setting used in the Kotlin mod (default: "jpg").
IMAGE_EXT: str = "jpg"

# Image size fed to the network - (H, W), rectangular to preserve the 16:9
# source ratio (512x288).  224x384 keeps mobs ~2x larger than the old 160x160
# square and is divisible by 32 (required by most backbones).
IMG_SIZE: tuple[int, int] = (224, 384)

# Number of bits per channel for RandomPosterize augmentation (train only).
# Set to None to disable.  4 = 16 shades per channel (reduces colour noise).
POSTERIZE_BITS: int | None = 4

# Equalise the V (Value) channel in HSV before training to neutralise
# Minecraft's 24-hour lighting cycle. Applied deterministically to train + val.
EQUALIZE_V: bool = True
