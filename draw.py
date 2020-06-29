import tkinter as tk
from tkinter import *
from PIL import Image
from PIL import ImageOps 
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageTk
import PIL
import numpy as np
import os


def draw():
    width = 280
    height = 280
    center = height//2
    white = (255, 255, 255)


    def save():
        filename = "image.png"
        image1.save(filename)

        img = Image.open('image.png')
        resized_img = img.resize((28, 28), Image.ANTIALIAS) 
        resized_img = ImageOps.invert(resized_img)
        resized_img = resized_img.convert('L').save('image.png')
        if (os.path.isfile('image.png')):
    	    print('Image saved.')
        root.destroy()

    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        cv.create_oval(x1, y1, x2, y2, fill="black",width=15)# 9
        draw.line([x1, y1, x2, y2],fill="black",width=15)# 9 
        

    root = tk.Tk()
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)
    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)
    button=Button(text="save",command=save)
    button.pack()
    root.mainloop()
