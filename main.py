from audiotool import atoolclass
from audionet import anetclass
# from commandnet import commandclass
import pynput
from pynput import keyboard, mouse
import time
import os
import shutil

# import everything from tkinter module
from tkinter import *

# import messagebox from tkinter module
import tkinter.messagebox

import easygui

import constants

counter =0

def popup_message(message):
    root = tkinter.Tk()
    root.withdraw()
    tkinter.messagebox.showinfo("Igor says", message)

def recordCommandSequence(filestring):
    # allevents=[]
    counter =0
    allevents =[]
    ctime = time.time()
    # The event listener will be running in this block
    with keyboard.Events() as kevents, mouse.Events() as mevents:

        while True:
            kevent = kevents.get(0.001)
            if kevent:
                kevent.key
                print('Received event {}'.format(kevent))
                allevents.append(kevent)
                allevents.append(time.time())
                counter += 1
                ctime = time.time()

            mevent = mevents.get(0.001)
            if mevent:
                try:
                    if mevent.button:
                        allevents.append(mevent)
                        allevents.append(time.time())
                    counter +=1
                    print(counter)
                    ctime = time.time()
                except:
                    pass

            if (time.time() - ctime)>1:
                break

    with open(constants.SEQUENCES_PATH+filestring+".dat", "w") as inputfile:
        for entry in allevents:
            inputfile.write(str(entry)+"\n")
        inputfile.close()

    # return allevents


# def dosth(events):
def replayCommandSequence(filestring):
    mController = pynput.mouse.Controller()
    kController = pynput.keyboard.Controller()
    with open(constants.SEQUENCES_PATH+filestring+".dat", "r") as stringfile:
        filelines = stringfile.readlines()
        stringfile.close()

    lasteventtime = float(filelines[1])

    for entry in filelines:
        # KEY PRESS events:
        if entry[0] == "P":
            x= entry.rstrip("')\n")
            y= x.lstrip("Press(key=")
            mKeyPressed= y[1:]
            kController.press(mKeyPressed)

        # KEY RELEASE events:
        elif entry[0] == "R":
            x= entry.rstrip("')\n")
            y= x.lstrip("Release(key=")
            mKeyReleased= y[1:]
            kController.press(mKeyReleased)

        # CLICK events:
        elif entry[0]=="C":
            x= entry.rstrip(")\n")
            x= x.split(",")
            x= [entry.split("=",1)[1] for entry in x]
            mController.move(int(x[0])-mController.position[0], int(x[1])-mController.position[1])

            if x[2]=="Button.right":
                print("Button.right")
                if x[3] == "False":
                    mController.release(mouse.Button.right)
                elif x[3] == "True":
                    mController.press(mouse.Button.right)

            elif x[2]=="Button.left":
                print("Button.left")
                if x[3] == "False":
                    mController.release(mouse.Button.left)
                elif x[3] == "True":
                    mController.press(mouse.Button.left)


            time.sleep(2)
            print (x)

        elif entry[0] !="":
            neweventtime = float(entry)
            while ((neweventtime-lasteventtime)>0.01):
                time.sleep(0.1)
                lasteventtime+=0.1
                print("ugh")
                print(float(entry)-lasteventtime)
                break
            lasteventtime = neweventtime
            print(entry)
        print("############################################################")

    return 0

def main():
    manet = anetclass()
    maudiotool = atoolclass()

    print("vorbereitungen abgeschlossen \n\n\n")
    igor_flag = "passiv"

    while True:

        maudiotool.listen()
        word_class = manet.classify("input_audio.wav")

        # ###################################################################
        # global counter
        # counter +=1
        # if counter<=1:
        #     word_class = "igor"
        # if counter >=3:
        #     break
        # if counter ==2:
        #     word_class= "record"
        # ###################################################################

        print(word_class)
        if igor_flag == "passiv":
            if word_class == "igor":
                igor_flag = "aktiv"
                print("igor_flag = True")
                # word_class = "blah"
                continue
            else:
                continue
            # igor_flag = True

        if igor_flag == "aktiv":
            if word_class == "record":

                new_word = easygui.enterbox(msg="", title=' ', default='', strip=True, image=None, root=None)
                
                for mpath in os.listdir(constants.COMMANDS_AUDIO_PATH):
                    if not os.path.isfile(mpath):
                        print(mpath)
                        if new_word == mpath:
                            igor_flag ="passiv"
                            popup_message("neues audiosample für commando "+ str(new_word)+" abgespeichert")    
                            newname= str(len(os.listdir(constants.COMMANDS_AUDIO_PATH+new_word)))
                            shutil.copy2("input_audio.wav", constants.COMMANDS_AUDIO_PATH+new_word+"/"+newname+".wav"  )
                            break                          
                if igor_flag=="passiv":
                    # bereits bekanntes Verzeichnes -> zurück zum Anfang
                    break
                
                # unbekanntes Verzeichnis -> neue Befehlssequenz                
                popup_message("beginnt aufnahme der commandosequence")
                recordCommandSequence(new_word)
                
                # new command dir
                for root, dirs, files in os.walk(constants.COMMANDS_AUDIO_PATH):
                    for dirname in dirs:
                        print(dirname)
                        if dirname[0]=="_":
                            print("a")
                            os.rename(constants.COMMANDS_AUDIO_PATH+dirname, constants.COMMANDS_AUDIO_PATH+new_word)
                            break
                shutil.copy2("input_audio.wav", constants.COMMANDS_AUDIO_PATH+new_word+"/0.wav"  )
                manet.train(manet.model)
                popup_message("igor hat was neues gelernt")
                igor_flag = "passiv"
                continue

            else:
                # igor führt erkannten befehl aus
                replayCommandSequence(word_class)
                igor_flag = "passiv"
                continue



if __name__ == "__main__":
    main()
    pass
