import os
import cairosvg
from PIL import Image
from svglib.svglib import svg2rlg
import matplotlib.pyplot as plt
from reportlab.graphics import renderPDF, renderPM
import cv2


def main():
    titles1 = ["Prediction loss", "Regularization loss"]
    titles2 = ["e_hotel", "e_univ", "e_zara1", "e_zara2"]
    folder = "./runs/E2_complex_beta_withoutz"
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        name = filename.split('.')[0] + '.png'
        out_path = os.path.join(folder, name)
        try:
            drawing = svg2rlg(path)
            renderPM.drawToFile(drawing, out_path, fmt="PNG")
        except:
            continue

    I = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            I.append(Image.open(path))
        except:
            continue

    plt.imshow(I[0])
    fig, axes = plt.subplots(1, 2)
    for i, (img, ax) in enumerate(zip(I[:2], axes)):
        ax.imshow(img)
        ax.set_title(titles1[i])

    fig.savefig('full_figure.png')

if __name__ == "__main__":
    main()
