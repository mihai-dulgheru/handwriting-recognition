import os

import pandas as pd
from pdf2image import convert_from_path

w = 7016
h = 4961
p = (300, 300, 430, 300)
no_lines = 16
no_columns = 10
w_cell = (w - p[3] - p[1]) / no_columns
h_cell = (h - p[0] - p[2]) / no_lines


def convert_pdf_to_image(pdf_path):
    return convert_from_path(
        pdf_path, 600, poppler_path="../../libs/poppler-0.68.0/bin"
    )


def crop_image(img):
    return img.crop((p[3], p[0], w - p[1], h - p[2]))


def write_to_file(lines):
    with open("../../data/words.txt", "a") as file:
        file.writelines([f"{line[0]} {line[1]}\n" for line in lines])


def generate_words(prefix, img, moves):
    lines = []
    for line in range(1, no_lines, 2):
        for column in range(no_columns):
            x = column * w_cell + 4
            y = line * h_cell + 4
            # .convert("1") or .convert("LA")
            # https://holypython.com/python-pil-tutorial/how-to-convert-an-image-to-black-white-in-python-pil/
            word = moves[column][int((line - 1) / 2)]
            filename = f"../../data/words/{prefix}-{word}.png"
            # Get the original file name and path
            original_file_name = os.path.basename(filename)
            original_file_path = os.path.dirname(filename)
            fp = os.path.join(original_file_path, original_file_name)
            img.crop((x, y, x + w_cell - 5, y + h_cell - 5)).convert("LA").save(
                fp=fp, format="PNG"
            )
            lines.append(
                (fp.replace("\\", "/").replace("../../", "../"), str(word).strip())
            )
            print(f"Image '{original_file_name}' has been successfully saved")
    write_to_file(lines)


if __name__ == "__main__":
    path = "chess_moves.pdf"
    images = convert_pdf_to_image(path)
    df = pd.read_excel("chess_moves.xlsx", index_col=None, header=None, comment="#")
    with open("../../data/words.txt", "w") as file:
        file.write(
            """#------------------------------ words.txt ------------------------------#
#
# format: ../data/words/ABC.png ABC
#
#     ../data/words/ABC.png     -> the path where the file is stored
#     ABC                       -> the transcription for this word
#
"""
        )
    j = 0
    for i in range(len(images)):
        generate_words(
            prefix=f"{chr(j + 97)}-{i + 1}", img=crop_image(images[i]), moves=df
        )
