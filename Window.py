import tkinter as tk
from tkinter.constants import *


# Create the canvas, size in pixels.
canvas = tk.Canvas(width=300, height=200, bg='black')

# Pack the canvas into the Frame.
canvas.pack(expand=YES, fill=BOTH)

# Load the .gif image file.
gif1 = tk.PhotoImage(file='./Original/Blue.png')

# Put gif image on canvas.
# Pic's upper-left corner (NW) on the canvas is at x=50 y=10.
canvas.create_image(0, 0, image=gif1, anchor=NW)

# Run it...
tk.mainloop()