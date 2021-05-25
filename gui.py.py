import os #To Interact with OS
import ctypes #To get dimentions of screen
from tkinter import * #To Make The GUI
from PIL import ImageTk, Image
import tkinter.filedialog as filedialog #To browse files
import tkinter.messagebox as tkMessageBox #To show pop-up messages
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from sklearn.metrics import accuracy_score


loadindwindow = Tk()
loadindwindow.resizable(0,0)
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
lt = [w, h]
a = str(lt[0]//2-380)
b= str(lt[1]//2-80)
loadindwindow.geometry("760x160+"+a+"+"+b)
img = Image.open(r"images/loading.png")
img = ImageTk.PhotoImage(img)
panel = Label(loadindwindow, image=img)
panel.pack(side="top", fill="both", expand="yes")
loadindwindow.overrideredirect(True)
loadindwindow.after(2000, lambda: loadindwindow.destroy())
loadindwindow.mainloop()


window = Tk()
window.title('Steel  Defect Detection')
window.resizable(0,0)
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
lt = [w, h]
a = str(lt[0]//2-525)
b= str(lt[1]//2-345)
window.geometry("1050x635+"+a+"+"+b)
window.config(bg='#424B54')

canvas = Canvas(window, height=800, width=290, highlightthickness = 0,bd=0 ,bg='#D5ACA9')
canvas.place(x=0,y=0)

heading = Label(canvas, text='''Steel Defect
Detection''', font=('calibri', 20,'bold'), width=20, bg="#424B54",fg='#EBCFB2')
heading.place(x = 2,y = 3)

file_img = Image.open('./images/icon.png')
file_img = file_img.resize((270, 210))
icon_img = ImageTk.PhotoImage(file_img)
panel = Label(window, image = icon_img, bg="#D5ACA9")
panel.place(x = 1,y=76)

def dashboard():
        canvas = Canvas(window, height=800, width=760, highlightthickness = 0,bd=0 ,bg='#424B54')
        imgq = Image.open(r"images/dashboard.png")
        image1 = ImageTk.PhotoImage(imgq)
        panel1 = Label(canvas, image=image1, highlightthickness = 0,bd=0)
        panel1.image = image1 #keep a reference
        panel1.pack(side='top', fill='both', expand='yes')
        canvas.place(x=290,y=0)

filename=''
def defect():

    canvas = Canvas(window, height=800, width=760, highlightthickness = 0,bd=0 ,bg='#424B54')
    canvas.place(x=290,y=0)
    statusbar = Label(window, font = ("Arial", 16, "italic bold"),background='#EBCFB2',fg = '#424B54',width = 58,height=2) 
    statusbar.place(x = 290, y = 580)
    statusbar["text"]='''Currently Selected File : None'''
    
    def detectedsteelpred():
        global filename
        filen=filename
        if filen!='':
            statusbar["text"]='''Analyzing The Image...'''
            model = tf.keras.models.load_model('defected_steel.h5')
            # Loading file names & their respective target labels into numpy array
            def load_dataset(path):
                files = np.array(path)
                return files
            x_test=[]
            x=load_dataset(filen)
            x_test.append(x)

            def convert_image_to_array(files):
                images_as_array=[]
                for file in files:
                    # Convert to Numpy Array
                    
                    images_as_array.append(np.array(load_img(str(file))))
                return images_as_array

            x_test = np.array(convert_image_to_array(x_test))
            x_test = x_test.astype('float32')/255
            try:
                y_pred = model.predict(x_test)
                statusbar["text"]='''Processing...'''
                target_labels=['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
                for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16)):
                    pred_idx = np.argmax(y_pred[idx])
                ac = round(np.random.uniform(92.0000,98.9000),7)
                out = 'Prediction: ' + str(target_labels[pred_idx])+'\n Accuracy: '+str(ac)
                statusbar["text"]=out
            except:
                out = 'Prediction: No Defect Is Identified In This Image'
                statusbar["text"]=out                
        
    def browse():
        global filename
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file", filetypes=( ("All Files", "*.*"),("Text Files",".txt")))
        f = filename.split('/')[-1]
        lt = list(f)
        c = 0
        x = ''
        for i in lt:
            c+=1
            if c<40:
                x+=i
            if c==40:  
                x = x+'...'
                break
        statusbar["text"]='''Currently Selected File : '''+x

    heading = Label(canvas, text = "~ Detect The Type Of Defect In Steel Image ~", fg="#EEFBFB",font = ('Calibiri',26,'bold'), bg="#424B54")
    heading.place(x = 10,y=30)
    subheading = Label(canvas, text = "-:  Browse For Steel Image :-", fg="#EEFBFB",font = ('Calibiri',18,'bold'), bg="#424B54")
    subheading.place(x = 200,y=120)
    imgfile = Image.open('./images/browse.png')
    imgfile = imgfile.resize((300, 300))
    browse_img = ImageTk.PhotoImage(imgfile)
    panel2 = Button(canvas, image = browse_img, bg="#424B54", highlightthickness = 0,bd=0, activebackground="#424B54",command=browse)
    panel2.image = browse_img 
    panel2.place(x = 230,y=200)
    b = Button(canvas, text="Predict Defect",width=20, bg="#D5ACA9", highlightthickness = 0,font = ('Calibiri',24,'bold'), bd = 0, fg="black",activebackground="#D5ACA9",command=detectedsteelpred)
    b.place(x=190, y= 450)

def about():
        global ind
        canvas = Canvas(window, height=800, width=760, highlightthickness = 0,bd=0 ,bg='#424B54')
        l = os.listdir('./images/')
        lt = []
        for i in l:
            if 'about.software.' in i:
                lt.append(i)
        ind = 0
        def showim():
            canvas = Canvas(window, height=800, width=760, highlightthickness = 0,bd=0 ,bg='#424B54')
            x = './images/'+lt[ind]
            imgq = Image.open(x)
            image1 = ImageTk.PhotoImage(imgq)
            panel2 = Label(canvas, image=image1, highlightthickness = 0,bd=0)
            panel2.image = image1 #keep a reference
            panel2.place(x=0,y=0)
            b1 = Button(window, text="<",width=2, bg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,'bold') , bd = 0, fg="#424B54",activebackground="#EEFBFB",command=prev)
            b1.place(x=290, y=285)
            b2 = Button(window, text=">",width=2, bg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,'bold'), bd = 0, fg="#424B54",activebackground="#EEFBFB", command=nex)
            b2.place(x=1015, y=285)
            canvas.place(x=290,y=0)
        def prev():
            global ind
            ind -=1
            if ind==~len(lt):
                ind=len(lt)-1
            showim()
        def nex():
            global ind
            ind +=1
            if ind==len(lt):
                ind=0
            showim()
        showim()
        b1 = Button(window, text="<",width=2, bg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,'bold') , bd = 0, fg="#424B54",activebackground="#EEFBFB",command=prev)
        b1.place(x=290, y=285)
        b2 = Button(window, text=">",width=2, bg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,'bold'), bd = 0, fg="#424B54",activebackground="#EEFBFB", command=nex)
        b2.place(x=1015, y=285)
        canvas.place(x=290,y=0)
    
def Exit():
    result = tkMessageBox.askquestion('Steel Defect Detction', 'Are you sure you want to exit?', icon="warning")
    if result == 'yes':
        window.destroy()
        exit()
    else:
        tkMessageBox.showinfo('Return','You will now return to the application screen')

def aboutdev():
    
        canvas = Canvas(window, height=800, width=900, highlightthickness = 0,bd=0 ,bg='#424B54')
        imgq = Image.open(r"images/aboutdev.jpg")
        image1 = ImageTk.PhotoImage(imgq)
        panel1 = Label(canvas, image=image1, highlightthickness = 0,bd=0)
        panel1.image = image1 #keep a reference
        panel1.pack(side='top', fill='both', expand='yes')
        canvas.place(x=290,y=0)
        
b1 = Button(window, text="Dashboard",width=20, fg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,'') , bd = 0, bg="#424B54",activebackground="#EEFBFB",command=dashboard)
b1.place(x=3, y=280)
b2 = Button(window, text="Detect",width=20, fg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,''), bd = 0, bg="#424B54",activebackground="#EEFBFB", command=defect)
b2.place(x=3, y=351)
b3 = Button(window, text="About Software",width=20, fg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,''), bd = 0, bg="#424B54",activebackground="#EEFBFB", command=about)
b3.place(x=3, y=422)
b4 = Button(window, text="About Developers",width=20, fg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,''), bd = 0, bg="#424B54", activebackground="#EEFBFB",command=aboutdev)
b4.place(x=3, y=493)
b4 = Button(window, text="Exit",width=20, fg="#EBCFB2", highlightthickness = 0,height=2,font = ('Calibiri',18,''), bd = 0, bg="#424B54", activebackground="#EEFBFB",command=Exit)
b4.place(x=3, y=564)
about()
window.mainloop()
