"""
Thanks ChatGPT for pairing.
"""

import glob
import argparse
import re
import math
import os
from PIL import Image, ImageDraw, ImageFont
from diffusers.utils import make_image_grid  # Assuming this is available


def add_text_to_image(image: Image.Image, text: str, position=(10, 10), color="ivory") -> Image.Image:
    """
    Draws the given text on the image at the specified position.
    """
    draw = ImageDraw.Draw(image)
    try:
        # Try using a commonly available TrueType font with the specified font size.
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except Exception as e:
        print(f"Could not load TrueType font: {e}. Falling back to default font.")
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    return image


def compute_grid(n):
    """
    Compute a grid layout (rows, cols) that is as square as possible.
    For n=1,2,3, it returns a single row and n columns.
    For example:
      - 4 videos -> 2 rows x 2 cols
      - 6 videos -> 2 rows x 3 cols
    """
    if n < 4:
        return 1, n
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def create_video_collage(video_files, output_filename="collage.mp4"):
    from moviepy import VideoFileClip, clips_array, ColorClip

    n = len(video_files)
    if n == 0:
        raise ValueError("No video files provided.")

    # Determine grid size automatically.
    rows, cols = compute_grid(n)
    print(f"Creating collage with grid: {rows} rows x {cols} columns")

    # Load all video clips.
    clips = [VideoFileClip(f) for f in video_files]

    # Determine a common size for all videos by using the smallest width and height.
    min_width = min(clip.w for clip in clips)
    min_height = min(clip.h for clip in clips)
    target_size = (min_width, min_height)
    clips = [clip.resized(target_size) for clip in clips]

    # If grid slots exceed the number of videos, fill the rest with black clips.
    total_slots = rows * cols
    if len(clips) < total_slots:
        # Use the minimum duration among the clips to create dummy clips.
        dummy_duration = min(clip.duration for clip in clips)
        blank_clip = ColorClip(size=target_size, color=(0, 0, 0), duration=dummy_duration)
        clips.extend([blank_clip] * (total_slots - len(clips)))

    # Arrange clips into a grid (list of lists) for clips_array.
    grid = []
    for i in range(rows):
        row_clips = clips[i * cols : (i + 1) * cols]
        grid.append(row_clips)

    # Create the collage using MoviePy's clips_array.
    collage = clips_array(grid)

    # Write the final video file.
    collage.write_videofile(output_filename, codec="libx264", audio_codec="aac")


def derive_collage_filename(prompt: str, sorted_filenames: list, is_mp4=False) -> str:
    """
    Derives a representative filename for the collage based on the group prompt and the range of i values.
    """
    # Use a regex to extract the i value from a filename (assuming the pattern _i@<number>_)
    i_pattern = re.compile(r"_i@(\d+)_")
    i_values = []
    extension_to_use = "mp4" if is_mp4 else "png"
    for fname in sorted_filenames:
        m = i_pattern.search(fname)
        if m:
            i_values.append(int(m.group(1)))
    if not i_values:
        return f"collage_{prompt}.{extension_to_use}"
    min_i, max_i = min(i_values), max(i_values)
    # Create a filename that shows the prompt and the i range.
    return f"collage_{prompt}_i@{min_i}-{max_i}.{extension_to_use}"


def main(args):
    # Get all JSON files in the current directory that include 'hash' in their name.
    json_files = glob.glob(f"{args.path}/*.json")
    json_files = [f for f in json_files if "hash" in f]
    assert json_files

    mp4s = glob.glob(f"{args.path}/*.mp4")
    has_mp4 = True if mp4s else False

    # Regular expression to extract prompt, hash, i, and seed.
    pattern = re.compile(r"prompt@(.+?)_hash@([^_]+)_i@(\d+)_s@(\d+)\.json")

    # Group files by their prompt.
    groups = {}
    for filename in json_files:
        match = pattern.search(filename)
        if match:
            prompt, file_hash, i_str, seed = match.groups()
            groups.setdefault(prompt, []).append(filename)

    print(f"Total groups found: {len(groups)}.")

    # Process each group separately.
    for prompt, files in groups.items():
        # Sort filenames in the group by the integer value of i i.e., the search.
        sorted_files = sorted(files, key=lambda fname: int(pattern.search(fname).group(3)))

        # Load corresponding PNG images and annotate them with the i value.
        annotated_images = []
        video_files = []
        for fname in sorted_files:
            # Extract the i value from the filename.
            i_val = int(pattern.search(fname).group(3))
            # Replace .json with .png to get the image filename.
            extension_to_use = ".mp4" if has_mp4 else ".png"
            png_filename = fname.replace(".json", extension_to_use)
            if not has_mp4:
                try:
                    image = Image.open(png_filename)
                except Exception as e:
                    print(f"Could not open image '{png_filename}': {e}")
                    continue

                # Annotate the image with "i=<value>" in the top-left corner.
                annotated_image = add_text_to_image(image, f"i={i_val}")
                annotated_images.append(annotated_image)
            else:
                video_files.append(png_filename)

        # Derive a representative collage filename.
        collage_filename = derive_collage_filename(prompt, sorted_files, is_mp4=has_mp4)
        collage_filename = os.path.join(args.path, collage_filename)

        if not has_mp4:
            if not annotated_images:
                print(f"No valid images for prompt '{prompt}'.")
                continue

            # Create a collage (horizontal grid: one row, all images as columns).
            grid = make_image_grid(annotated_images, rows=1, cols=len(annotated_images))

            grid.save(collage_filename)
            print(f"Saved collage for prompt '{prompt}' as '{collage_filename}'.")
        else:
            if not video_files:
                print(f"No valid videos for prompt '{prompt}'.")
                continue
            create_video_collage(video_files, output_filename=collage_filename)
            print(f"Saved collage for prompt '{prompt}' as '{collage_filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path containing the JSON AND image files.")
    args = parser.parse_args()
    main(args)
