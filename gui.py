from tkinter import *

def draw_gui(model):
    top = Tk()
    top.geometry("400x250")
    user_sentence_label = Label(top, text="Enter your sentence").place(x=30, y=50)

    sbmitbtn = Button(top, text="Submit", activebackground="pink", activeforeground="blue").place(x=30, y=170)

    user_entry = Entry(top).place(x=80, y=50)

    var = StringVar()
    label = Message(top, textvariable=var, relief=RAISED)

    var.set("Hey!? How are you doing?")
    label.pack()

    top.mainloop()