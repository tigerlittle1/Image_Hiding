import tkinter as tk
import tkinter.messagebox
from tkinter import ttk , Checkbutton
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

from tkinter import messagebox
from Hiding import Hiding_Lin_method,Hiding_modify_method,Hiding_quaternary_method,Hiding_base25_method
import numpy as np
import threading
import argparse

class Hiding_windows():
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("800x500")
        self.window.title("Hiding_windows")
        self.top_frame = tk.Frame(self.window)
        self.top_frame.pack()

        self.cover_img_label = tk.Label(self.top_frame)
        self.cover_img_label.grid(column=0, row=0, pady=50, padx=50)
        self.Button1 = tk.Button(self.top_frame, text="select file",command = self.seelect_coverimage)
        self.Button1.grid(column=0, row=1)

        self.screat_img_label = tk.Label(self.top_frame)
        self.screat_img_label.grid(column=0, row=2, pady=50, padx=50)
        self.Button2 = tk.Button(self.top_frame, text="select file",command = self.seelect_screatimage)
        self.Button2.grid(column=0, row=3)

        self.combo = ttk.Combobox(self.top_frame,
                                    values=[
                                        "Hiding_quaternary_method",
                                        "Hiding_Lin_method",
                                        "Hiding_modify_method",
                                        "Hiding_base25_method"])
        self.combo.current(0)
        self.combo.grid(column=1, row=0, columnspan=1, rowspan=4)
        self.Button3 = tk.Button(self.top_frame, text="Hidding",command =self.hid_image)
        self.Button3.grid(column=1, row=3, pady=50, padx=50)

        self.stego_image_label = tk.Label(self.top_frame)
        self.stego_image_label.grid(column=2, row=0, pady=50, padx=50, columnspan=2, rowspan=4)
        self.Button4 = tk.Button(self.top_frame, text="Save",command = self.save_key_and_stego_image)
        self.Button4.grid(column=2, row=3, pady=50, padx=50)

        self.cover_image = None
        self.screat_image = None
        self.stego_image = None

        self.hiding_system = None

        self.parallel = tk.BooleanVar()
        self.parallel.set(True)
        self.parallel_button = Checkbutton(self.top_frame, text="Use parallel", variable=self.parallel,
                         onvalue=True, offvalue=False, height=5,
                         width=20)
        self.parallel_button.grid(column=1, row=1, columnspan=1, rowspan=4)

    def file_path(self):
        file_path = filedialog.askopenfilename(filetypes = (("image files",("*.png","*.jpg")),("all files","*.*")))
        return file_path

    def cv2_to_tkimage(self,image):
        if image.shape[2] > 1:
            b, g, r = cv2.split(image)[0:3]
            image = cv2.merge((r, g, b))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)
        return image

    def seelect_coverimage(self):
        file_path = self.file_path()
        self.cover_image = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        shape = self.cover_image.shape
        image = cv2.resize(self.cover_image, (int(shape[1]/5),int(shape[0]/5)))
        image = self.cv2_to_tkimage(image)
        self.cover_img_label.config(image = image)
        self.cover_img_label.image = image

    def seelect_screatimage(self):
        file_path = self.file_path()
        self.screat_image = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        shape = self.screat_image.shape
        image = cv2.resize(self.screat_image, (int(shape[1]/5),int(shape[0]/5)))
        image = self.cv2_to_tkimage(image)
        self.screat_img_label.config(image = image)
        self.screat_img_label.image = image

    def hidding_thread(self):
        self.stego_image = self.hiding_system.hiding_message_parallel(self.cover_image , self.screat_image)

        if type(self.stego_image) == str:
            tkinter.messagebox.showerror("hiding error", "The secret message is too long , please use larger imag")
        else:
            shape = self.stego_image.shape
            img = cv2.resize(self.stego_image, (int(shape[1]/5),int(shape[0]/5)))
            img = self.cv2_to_tkimage(img)
            self.stego_image_label.config(image =  img)
            self.stego_image_label.image = img
        self.Button1.config(state = tk.NORMAL)
        self.Button2.config(state=tk.NORMAL)
        self.Button3.config(state=tk.NORMAL)
        self.Button4.config(state=tk.NORMAL)
        print("PSNR : ",self.hiding_system.caculate_PSNR(self.cover_image,self.stego_image))

    def hid_image(self):
        if self.cover_image is None or self.screat_image is None:
            tkinter.messagebox.showerror("select file error", "please select cover image and screat image")
        else:
            self.hiding_system = globals()['{}'.format( self.combo.get())](self.parallel.get())

            t = threading.Thread(target = self.hidding_thread)
            t.start()
            self.Button1.config(state = tk.DISABLED)
            self.Button2.config(state=tk.DISABLED)
            self.Button3.config(state=tk.DISABLED)
            self.Button4.config(state=tk.DISABLED)

    def save_file_path(self,filename):
        file_path = filedialog.asksaveasfilename(initialfile=filename ,filetypes = (("png files","*.png"),("all files","*.*")))
        return file_path

    def save_key_and_stego_image(self):
        if self.stego_image is None:
            tkinter.messagebox.showerror("select file error", "you didn't have stego image")
        else:
            path = self.save_file_path(filename = "stego_image.png")
            cv2.imencode('.png', self.stego_image)[1].tofile(path)
            self.hiding_system.save_key(path[:-4] + ".npz")
            tkinter.messagebox.showinfo('Save image and key', 'your stego image and key is save as '+path)

class Extracting_windows():
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("800x500")
        self.window.title("Extracting_windows")
        self.top_frame = tk.Frame(self.window)
        self.top_frame.pack()

        self.stego_image_label = tk.Label(self.top_frame,text="stego_image")
        self.stego_image_label.grid(column=0, row=0, pady=50, padx=50)
        self.Button1 = tk.Button(self.top_frame, text="select stego image file",command = self.select_stego_image)
        self.Button1.grid(column=0, row=1)

        self.key_label = tk.Label(self.top_frame,text = "key file")
        self.key_label.grid(column=0, row=2, pady=50, padx=50)
        self.Button2 = tk.Button(self.top_frame, text="select key file",command = self.select_key)
        self.Button2.grid(column=0, row=3)

        self.screat_img_label = tk.Label(self.top_frame,text="screat_img")
        self.screat_img_label.grid(column=3, row=2)

        self.combo = ttk.Combobox(self.top_frame,
                                    values=[
                                        "Hiding_quaternary_method",
                                        "Hiding_Lin_method",
                                        "Hiding_modify_method",
                                        "Hiding_base25_method"])

        self.combo.current(0)
        self.combo.grid(column=1, row=0, columnspan=1, rowspan=4)
        self.Button3 = tk.Button(self.top_frame, text="Extracting",command = self.extracting_image)
        self.Button3.grid(column=1, row=3, pady=100, padx=100)

        self.Button4 = tk.Button(self.top_frame, text="Save screat image",command = self.Save_screat_image)
        self.Button4.grid(column=3, row=3, pady=100, padx=100)

        self.screat_image = None
        self.stego_image = None
        self.hiding_system = None

        self.parallel = tk.BooleanVar()
        self.parallel.set(False)
        self.parallel_button = Checkbutton(self.top_frame, text="Use parallel", variable=self.parallel,
                         onvalue=True, offvalue=False, height=5,
                         width=20)
        self.parallel_button.grid(column=1, row=1, columnspan=1, rowspan=4)

    def file_path(self,file_type = ("image files","*.png")):
        file_path = filedialog.askopenfilename(filetypes = (file_type,("all files","*.*")))
        return file_path

    def cv2_to_tkimage(self,image):
        b, g, r = cv2.split(image)
        image = cv2.merge((r, g, b))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)
        return image

    def select_stego_image(self):
        file_path = self.file_path()
        self.stego_image = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        shape = self.stego_image.shape
        image = cv2.resize(self.stego_image, (int(shape[1]/5),int(shape[0]/5)))
        image = self.cv2_to_tkimage(image)
        self.stego_image_label.config(image = image)
        self.stego_image_label.image = image

    def select_key(self):
        file_path = self.file_path(("key files","*.npz"))
        self.key_label.config(text = file_path)
        self.key_label.text = file_path

    def extracting_thread(self):
        try:
            self.screat_image = self.hiding_system.extracting_message_parallel(self.stego_image,self.key_label.text)

            if type(self.screat_image) == str:
                tkinter.messagebox.showerror("extracting error", "extracting error")
            else:
                shape =self.screat_image.shape
                img = cv2.resize(self.screat_image, (int(shape[1]/5),int(shape[0]/5)))
                # cv2.imshow("result",self.screat_image)
                # cv2.waitKey()
                img = self.cv2_to_tkimage(img)
                self.screat_img_label.config(image =  img)
                self.screat_img_label.image = img
        except:
            tkinter.messagebox.showerror("extracting file error", "please select true method")
        self.Button1.config(state = tk.NORMAL)
        self.Button2.config(state=tk.NORMAL)
        self.Button3.config(state=tk.NORMAL)
        self.Button4.config(state=tk.NORMAL)

    def extracting_image(self):

        if self.stego_image is None or ".npz" not in self.key_label.text :
            tkinter.messagebox.showerror("select file error", "please select stego image and key file")
        else:
            # if self.combo.get() == "Hiding_Kim_method":
            #     print("Hiding_Kim_method")
            #     self.hiding_system = Hiding_Kim_method()
            # else:
            #     print("Hiding_modify_method")
            #     self.hiding_system = Hiding_modify_method()
            self.hiding_system = globals()['{}'.format(self.combo.get())](self.parallel.get())

            t = threading.Thread(target = self.extracting_thread)
            t.start()

        self.Button1.config(state = tk.DISABLED)
        self.Button2.config(state=tk.DISABLED)
        self.Button3.config(state=tk.DISABLED)
        self.Button4.config(state=tk.DISABLED)

    def save_file_path(self,filename):
        file_path = filedialog.asksaveasfilename(initialfile=filename ,filetypes = (("png files","*.png"),("all files","*.*")))
        return file_path

    def Save_screat_image(self):
        if self.screat_image is None:
            tkinter.messagebox.showerror("select file error", "you didn't have stego image")
        else:
            path = self.save_file_path("screat_image.png")
            # print(path)
            cv2.imencode('.png', self.screat_image)[1].tofile(path)
            tkinter.messagebox.showinfo('Save screat image', 'your screat image is save as ' + path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hiding system')
    parser.add_argument('--mode', type=str, default="ext",
                        help='mode of Hiding system (hid or ext)')
    args = parser.parse_args()

    if args.mode == "hid" :
        windows = Hiding_windows()
        windows.window.mainloop()
    elif args.mode == "ext"  :
        windows = Extracting_windows()
        windows.window.mainloop()
    else:
        print("please select hid or ext")