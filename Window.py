import tkinter as tk
from tkinter.constants import *

from CandyCrushGym import *

def main():

    root_img_path = "./Images"
        
    env = CandyCrushGym()

    state = env.reset()

    image_size = 60
    canvas = tk.Canvas(width=env.FIELD_SIZE * image_size, height=env.FIELD_SIZE * image_size, bg='black')
    canvas.pack(expand=YES, fill=BOTH)
    
    
    images = []
    for y in range(env.FIELD_SIZE):
        for x in range(env.FIELD_SIZE):
            candyID = state[y,x]

            if env.isNormalCandy(candyID):
                file_name = convert_normalCandyID_name(candyID)
                image = tk.PhotoImage(file=f"{root_img_path}/Normal/{file_name}.png")
            
            elif env.isWrappedCandyID(candyID):
                candyID = env.convertWrappedCandy_toNormal(candyID)
                file_name = convert_normalCandyID_name(candyID)
                image = tk.PhotoImage(file=f"{root_img_path}/Wrapped/{file_name}.png") 
            
            elif env.isHorizontalStrippedCandy(candyID):
                candyID = env.convertHorizontalStrippedCandy_toNormal(candyID)
                file_name = convert_normalCandyID_name(candyID)
                image = tk.PhotoImage(file=f"{root_img_path}/Striped/Horizontal/{file_name}.png")

            elif env.isVerticalStrippedCandy(candyID):
                candyID = env.convertVerticalStrippedCandy_toNormal(candyID)
                file_name = convert_normalCandyID_name(candyID)
                image = tk.PhotoImage(file=f"{root_img_path}/Striped/Vertical/{file_name}.png")

            images.append(image)
            canvas.create_image(x*image_size, y*image_size, image=image, anchor=NW)


    tk.mainloop()


def convert_normalCandyID_name(candyID):
    if candyID == 1:
        return "Red"
    elif candyID == 2:
        return "Orange"
    elif candyID == 3:
        return "Yellow"
    elif candyID == 4:
        return "Green"
    elif candyID == 5:
        return "Blue"
    elif candyID == 6:
        return "Purple"

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
