"""
Patch to add vertical flip option for CT training
Add this to base_dataset.py if you want vertical flipping
"""

# Add to get_params function:
def get_params_with_vertical(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip_h = random.random() > 0.5  # horizontal flip
    flip_v = random.random() > 0.5 if hasattr(opt, 'vertical_flip') and opt.vertical_flip else False

    return {"crop_pos": (x, y), "flip": flip_h, "flip_v": flip_v}

# Add vertical flip transform:
def __flip_vertical(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)