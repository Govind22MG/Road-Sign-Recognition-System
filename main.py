import os
import cv2
import numpy as np
from tkinter import *
from tkinter import font
from tkinter import filedialog
from PIL import Image, ImageTk
import rsb_recognizer

rsb_prediction = "None"

screen = Tk()
screen.geometry("960x480")
screen.title("Road Sign Board Classifier")
screen.configure(bg="#000000")

def browse_button_callback():
    global rsb_prediction
    filename = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select an 80x80 image file",
                                          filetypes = (("all files", "*.*"),
                                                       ("Text files", "*.txt*")))
    if(os.path.exists(filename)):
        input_img = cv2.imread(filename)
        rsb_prediction = rsb_recognizer.predict(input_img)
        input_img = cv2.resize(input_img, (640, 480))
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image = img)
        tk_img_frame.imgtk = imgtk
        tk_img_frame.configure(image=imgtk)

tk_img_frame = Label(screen)
tk_img_frame.grid(row=0, column=0)

tk_prediction_label = Label(screen,
                            text="Pred: " + rsb_prediction,
                            bg="#000000",
                            fg="#FFFFFF",
                            font=('Segoe UI Semibold', 20))
tk_prediction_label.place(x=670, y=20)

browse_button_obj = Button(screen,
                           text ="Browse Image",
                           bg="#FFFFFF",
                           fg="#000000",
                           font=('Segoe UI Semibold', 16),
                           command = browse_button_callback)
browse_button_obj.place(x=710, y=100)

def tk_show_update_screen():
    global rsb_prediction
    tk_prediction_label.configure(text="Pred : " + rsb_prediction)
    tk_img_frame.after(50, tk_show_update_screen)

#browse_button_obj.pack()
tk_show_update_screen()
screen.mainloop()


