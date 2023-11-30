import matplotlib.pyplot as plt

def bresenham_circle(cmin, cmax, Nx, Ny):
  r = 0.5 # nm
  y = 0
  err = 0

  points = []

  while r >= y:
    points.append((Nx + r, Ny + y))
    points.append((Nx + y, Ny + r))
    points.append((Nx - y, Ny + r))
    points.append((Nx - r, Ny + y))
    points.append((Nx - r, Ny - y))
    points.append((Nx - y, Ny - r))
    points.append((Nx + y, Ny - r))
    points.append((Nx + r, Ny - y))
    y += 0.01
    err += 1 + 2*y
    if 2*(err-r) + 1 > 0:
      r -= 1
      err += 1 - 2*r

  return points

points = bresenham_circle(-1, 1, 50, 50)

# plot the points
plt.scatter([r for r, y in points], [y for r, y in points])

# draw the circle
circle = plt.Circle((50, 50), 30, fill=False)
plt.gcf().gca().add_artist(circle)

plt.show()