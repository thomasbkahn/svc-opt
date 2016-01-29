def cm_to_bokeh(cm):
    colors = [rgb2hex(cm(n)) for n in range(256)]
    return colors

def rgb2hex(tup):
    r, g, b, a = tup
    r = int(255*r)
    g = int(255*g)
    b = int(255*b)
    hex_str = "#{0:02x}{1:02x}{2:02x}".format(r, g, b)
    return hex_str

def get_color_index(val, vmin=0.0, vmax=1.0):
    if val <= vmin:
        return 0
    elif val >= vmax:
        return 255
    else:
        return int((val - vmin)/(vmax - vmin)*255)