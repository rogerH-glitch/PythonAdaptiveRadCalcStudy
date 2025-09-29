import numpy as np
import matplotlib.pyplot as plt

print("Hello Fire Engineer! Your setup works.")

# Make a simple 2D plot of two rectangles
rect1_x = [0, 1, 1, 0, 0]
rect1_y = [0, 0, 1, 1, 0]

rect2_x = [0, 1, 1, 0, 0]
rect2_y = [2, 2, 3, 3, 2]

plt.plot(rect1_x, rect1_y, 'b-', label="Rectangle 1")
plt.plot(rect2_x, rect2_y, 'r-', label="Rectangle 2")
plt.legend()
plt.axis("equal")
plt.show()
