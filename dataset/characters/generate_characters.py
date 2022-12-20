import os

from pdf2image import convert_from_path

no_lines = 6
no_columns = 8
w_cell = 541
h_cell = 260
p = (300, 300)
b = 4


def convert_pdf_to_image(pdf_path):
    return convert_from_path(
        pdf_path, 600, poppler_path="../../libs/poppler-0.68.0/bin"
    )


def generate_characters(prefix, img, characters):
    for line in range(1, no_lines, 2):
        for column in range(no_columns):
            x = p[1] + column * (w_cell + b) + b
            y = p[0] + line * (h_cell + b) + b
            # .convert("1") or .convert("LA")
            # https://holypython.com/python-pil-tutorial/how-to-convert-an-image-to-black-white-in-python-pil/
            filename = f"../../data/characters/{prefix}-{characters[int((line - 1) / 2) * no_columns + column]}.png"
            # Get the original file name and path
            original_file_name = os.path.basename(filename)
            original_file_path = os.path.dirname(filename)
            fp = os.path.join(original_file_path, original_file_name)
            img.crop((x, y, x + w_cell, y + h_cell)).convert("LA").save(
                fp=fp, format="PNG"
            )
            print(f"Image '{original_file_name}' has been successfully saved")


if __name__ == "__main__":
    images = convert_pdf_to_image("characters.pdf")
    chars = [
        "R",
        "D",
        "T",
        "N",
        "C",
        "a",
        "b",
        "c ",
        "d ",
        "e",
        "f",
        "g",
        "h",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        "O",
        "-",
        "+",
    ]
    j = 0
    for i in range(len(images)):
        generate_characters(
            prefix=f"{chr(j + 97)}-{i + 1}", img=images[i], characters=chars
        )
