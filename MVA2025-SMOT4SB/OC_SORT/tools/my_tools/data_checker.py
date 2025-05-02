import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageNavigator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Navigator")
        
        self.image_list = []
        self.current_index = 0
        
        self.label = tk.Label(root)
        self.label.pack()
        
        self.btn_prev = tk.Button(root, text="Previous", command=self.show_prev)
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        
        self.btn_next = tk.Button(root, text="Next", command=self.show_next)
        self.btn_next.pack(side=tk.RIGHT, padx=10)
        
        self.load_images()
        self.show_image()
    
    def load_images(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            print("No folder selected.")
            self.root.quit()
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) 
                           if os.path.splitext(f)[1].lower() in valid_extensions]
        
        if not self.image_list:
            print("No valid images found.")
            self.root.quit()
    
    def show_image(self):
        if not self.image_list:
            return

        image_path = self.image_list[self.current_index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # 取得 Tkinter 視窗大小
        window_width = 800
        window_height = 600

        # 設定最大寬高，並保持圖片比例
        max_size = (window_width - 50, window_height - 100)  # 預留按鈕空間
        image.thumbnail(max_size, Image.Resampling.LANCZOS)  # 等比例縮小

        image = ImageTk.PhotoImage(image)
        
        self.label.config(image=image)
        self.label.image = image

    
    def show_next(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_image()
    
    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageNavigator(root)
    root.mainloop()
