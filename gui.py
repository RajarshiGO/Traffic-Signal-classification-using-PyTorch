#!usr/bin/python
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import torch
from torchvision import transforms
from module import CNN_model
#load the trained model to classify sign
model = CNN_model()
model.load_state_dict(torch.load('./parameters.pth'))
model.eval()
#dictionary to label all traffic signs class.
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            12:'Speed limit (50km/h)', 
            23:'Speed limit (60km/h)', 
            34:'Speed limit (70km/h)', 
            38:'Speed limit (80km/h)', 
            39:'End of speed limit (80km/h)', 
            40:'Speed limit (100km/h)', 
            41:'Speed limit (120km/h)', 
            42:'No passing', 
            2:'No passing veh over 3.5 tons', 
            3:'Right-of-way at intersection', 
            4:'Priority road', 
            5:'Yield', 
            6:'Stop', 
            7:'No vehicles', 
            8:'Veh > 3.5 tons prohibited', 
            9:'No entry', 
            10:'General caution', 
            11:'Dangerous curve left', 
            13:'Dangerous curve right', 
            14:'Double curve', 
            15:'Bumpy road', 
            16:'Slippery road', 
            17:'Road narrows on the right', 
            18:'Road work', 
            19:'Traffic signals', 
            20:'Pedestrians', 
            21:'Children crossing', 
            22:'Bicycles crossing', 
            24:'Beware of ice/snow',
            25:'Wild animals crossing', 
            26:'End speed + passing limits', 
            27:'Turn right ahead', 
            28:'Turn Left ahead', 
            29:'Ahead only', 
            30:'Go straight or right', 
            31:'Go straight or left', 
            32:'Keep right', 
            33:'Keep left', 
            35:'Roundabout mandatory', 
            36:'End of no passing', 
            37:'End no passing veh > 3.5 tons' }
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    transform = transforms.Compose([transforms.Resize((30, 30)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image = transform(image)
    image = image.view(1, 3, 30, 30)
    pred = model(image)
    _, lab = torch.max(pred, 1)
    print(lab.item())
    sign = classes[(lab.item())]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/5.25),(top.winfo_height()/5.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()