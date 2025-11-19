from PIL import Image


def rgba_to_rgb_white(input_path):
    """
    Convert an RGBA image to RGB with a white background.
    """
    img = Image.open(input_path).convert("RGBA")
    
    # Create white background
    background = Image.new("RGB", img.size, (255, 255, 255))
    
    # Paste RGBA on top, using alpha channel as mask
    background.paste(img, mask=img.split()[3])  # 3 = alpha channel

    return background

def combine_to_fixed_canvas(img1_path, img2_path, output_path, canvas_size=(1024, 512), direction="horizontal"):
    """
    Put two images into a fixed canvas, resizing each proportionally
    to occupy half of the canvas, with possible blank padding.
    """
    canvas_w, canvas_h = canvas_size

    # Load images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    if img1.mode != "RGB":
        img1= rgba_to_rgb_white(img1_path)

    if img2.mode != "RGB":
        img2 = rgba_to_rgb_white(img2_path)
    
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    if direction == "horizontal":
        # Each image gets half the width
        target_w, target_h = canvas_w // 2, canvas_h
    else:
        # Each image gets half the height
        target_w, target_h = canvas_w, canvas_h // 2

    # Resize both proportionally to fit into (target_w, target_h)
    img1.thumbnail((target_w, target_h), Image.LANCZOS)
    img2.thumbnail((target_w, target_h), Image.LANCZOS)

    # Create blank canvas
    combined = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    if direction == "horizontal":
        # Center img1 in left half
        offset_x1 = (target_w - img1.width) // 2
        offset_y1 = (target_h - img1.height) // 2
        combined.paste(img1, (offset_x1, offset_y1))

        # Center img2 in right half
        offset_x2 = target_w + (target_w - img2.width) // 2
        offset_y2 = (target_h - img2.height) // 2
        combined.paste(img2, (offset_x2, offset_y2))

    else:  # vertical
        # Center img1 in top half
        offset_x1 = (target_w - img1.width) // 2
        offset_y1 = (target_h - img1.height) // 2
        combined.paste(img1, (offset_x1, offset_y1))

        # Center img2 in bottom half
        offset_x2 = (target_w - img2.width) // 2
        offset_y2 = target_h + (target_h - img2.height) // 2
        combined.paste(img2, (offset_x2, offset_y2))

    combined.save(output_path)
    print(f"✅ Saved combined image at {output_path}")


# # Example usage
# combine_to_larger("image1.jpg", "image2.jpg", "combined.jpg", direction="horizontal")


from PIL import Image

def combine_foregrounds_background(foreground_paths, background_path, output_path, canvas_size=(1024, 512)):
    """
    Place multiple foregrounds on the left half (stacked vertically)
    and a background image on the right half, all resized proportionally
    to occupy as much space as possible in a fixed-size canvas.
    """
    canvas_w, canvas_h = canvas_size
    left_w, right_w = canvas_w // 2, canvas_w // 2  # left/right halves

    # Load background
    bg = Image.open(background_path).convert("RGB")
    bg= bg.resize((1280, 720))
    bg.thumbnail((right_w, canvas_h), Image.LANCZOS)

    # Prepare blank canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # Place background on right half, centered vertically
    offset_x = left_w + (right_w - bg.width) // 2
    offset_y = (canvas_h - bg.height) // 2
    canvas.paste(bg, (offset_x, offset_y))

    # Load and stack foregrounds in left half
    n = len(foreground_paths)
    target_h = canvas_h // n  # each foreground gets equal height
    for i, fpath in enumerate(foreground_paths):
        fg = Image.open(fpath)

        if fg.mode != "RGB":
            fg= rgba_to_rgb_white(fpath)

        fg = fg.convert("RGB")

        fg.thumbnail((left_w, target_h), Image.LANCZOS)

        # Center each foreground in its slot
        offset_x = (left_w - fg.width) // 2
        offset_y = i * target_h + (target_h - fg.height) // 2
        canvas.paste(fg, (offset_x, offset_y))

    # Save final image
    canvas.save(output_path)
    print(f"✅ Combined image saved at {output_path}")


# Example usage:
foregrounds = [
    # './Data/fun_material1/object_0.png',
    # './Data/fun_material1/object_1.png',
    # './Data/fun_material1/object_2.png'
    './Data/fun_material1/object_2.png',
    './Data/fun_material1/object_4.png'
]

# background = f"./tests/{idx}/main_entity.jpeg"
background = './Data/fun_material1/background.jpg'
# background = f"./tests/{idx}/main_entity.png"
combine_foregrounds_background(foregrounds, background, "./Data/fun2.png", canvas_size=(1280, 720))
