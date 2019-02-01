from collections import defaultdict
from math import sin, cos, pi, exp
import colorsys, base64, sys, os, math
import numpy as np
from scipy import misc
from xml.sax import saxutils

try:
  from cStringIO import StringIO as bytes_io
except ImportError:
  from io import BytesIO as bytes_io

TAU = 2*pi
LINE_TYPE = 'line'
SCATTER_TYPE = 'scatter'
BAR_TYPE = 'histogram'
DEFAULT_COLORS = ['#800000','#000080',
                  '#008000','#808000',
                  '#800080','#008080',
                  '#808080','#000000',
                  '#804000','#004080']

# Amino acid composition in the UniProtKB/Swiss-Prot data bank. Author(s): Bairoch A.
# Reference: Release notes for UniProtKB/Swiss-Prot release 2013_04 - April 2013. 
AMINO_ACID_PROP = {'A':0.0825,'L':0.0966,'Q':0.0393,'S':0.0656,
                   'R':0.0553,'K':0.0584,'E':0.0675,'T':0.0534,
                   'N':0.0406,'M':0.0242,'G':0.0707,'W':0.0108,
                   'D':0.0545,'F':0.0386,'H':0.0227,'Y':0.0292,
                   'C':0.0137,'P':0.0470,'I':0.0596,'V':0.0687}
  
# BLOSUM62 sunbstitution matrix with entries for X and O (hydroxyproline assumed to be same as proline)
BLOSUM62 = {'A':{'A': 4,'R':-1,'N':-2,'D':-2,'C': 0,'Q':-1,'E':-1,'G': 0,'H':-2,'I':-1,
                 'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S': 1,'T': 0,'W':-3,'Y':-2,'V': 0,'X':0},
            'R':{'A':-1,'R': 5,'N': 0,'D':-2,'C':-3,'Q': 1,'E': 0,'G':-2,'H': 0,'I':-3,
                 'L':-2,'K': 2,'M':-1,'F':-3,'P':-2,'S':-1,'T':-1,'W':-3,'Y':-2,'V':-3,'X':0},
            'N':{'A':-2,'R': 0,'N': 6,'D': 1,'C':-3,'Q': 0,'E': 0,'G': 0,'H': 1,'I':-3,
                 'L':-3,'K': 0,'M':-2,'F':-3,'P':-2,'S': 1,'T': 0,'W':-4,'Y':-2,'V':-3,'X':0},
            'D':{'A':-2,'R':-2,'N': 1,'D': 6,'C':-3,'Q': 0,'E': 2,'G':-1,'H':-1,'I':-3,
                 'L':-4,'K':-1,'M':-3,'F':-3,'P':-1,'S': 0,'T':-1,'W':-4,'Y':-3,'V':-3,'X':0},
            'C':{'A': 0,'R':-3,'N':-3,'D':-3,'C': 9,'Q':-3,'E':-4,'G':-3,'H':-3,'I':-1,
                 'L':-1,'K':-3,'M':-1,'F':-2,'P':-3,'S':-1,'T':-1,'W':-2,'Y':-2,'V':-1,'X':0},
            'Q':{'A':-1,'R': 1,'N': 0,'D': 0,'C':-3,'Q': 5,'E': 2,'G':-2,'H': 0,'I':-3,
                 'L':-2,'K': 1,'M': 0,'F':-3,'P':-1,'S': 0,'T':-1,'W':-2,'Y':-1,'V':-2,'X':0},
            'E':{'A':-1,'R': 0,'N': 0,'D': 2,'C':-4,'Q': 2,'E': 5,'G':-2,'H': 0,'I':-3,
                 'L':-3,'K': 1,'M':-2,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2,'X':0},
            'G':{'A': 0,'R':-2,'N': 0,'D':-1,'C':-3,'Q':-2,'E':-2,'G': 6,'H':-2,'I':-4,
                 'L':-4,'K':-2,'M':-3,'F':-3,'P':-2,'S': 0,'T':-2,'W':-2,'Y':-3,'V':-3,'X':0},
            'H':{'A':-2,'R': 0,'N': 1,'D':-1,'C':-3,'Q': 0,'E': 0,'G':-2,'H': 8,'I':-3,
                 'L':-3,'K':-1,'M':-2,'F':-1,'P':-2,'S':-1,'T':-2,'W':-2,'Y': 2,'V':-3,'X':0},
            'I':{'A':-1,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-3,'E':-3,'G':-4,'H':-3,'I': 4,
                 'L': 2,'K':-3,'M': 1,'F': 0,'P':-3,'S':-2,'T':-1,'W':-3,'Y':-1,'V': 3,'X':0},
            'L':{'A':-1,'R':-2,'N':-3,'D':-4,'C':-1,'Q':-2,'E':-3,'G':-4,'H':-3,'I': 2,
                 'L': 4,'K':-2,'M': 2,'F': 0,'P':-3,'S':-2,'T':-1,'W':-2,'Y':-1,'V': 1,'X':0},
            'K':{'A':-1,'R': 2,'N': 0,'D':-1,'C':-3,'Q': 1,'E': 1,'G':-2,'H':-1,'I':-3,
                 'L':-2,'K': 5,'M':-1,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2,'X':0},
            'M':{'A':-1,'R':-1,'N':-2,'D':-3,'C':-1,'Q': 0,'E':-2,'G':-3,'H':-2,'I': 1,
                 'L': 2,'K':-1,'M': 5,'F': 0,'P':-2,'S':-1,'T':-1,'W':-1,'Y':-1,'V': 1,'X':0},
            'F':{'A':-2,'R':-3,'N':-3,'D':-3,'C':-2,'Q':-3,'E':-3,'G':-3,'H':-1,'I': 0,
                 'L': 0,'K':-3,'M': 0,'F': 6,'P':-4,'S':-2,'T':-2,'W': 1,'Y': 3,'V':-1,'X':0},
            'P':{'A':-1,'R':-2,'N':-2,'D':-1,'C':-3,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-3,
                 'L':-3,'K':-1,'M':-2,'F':-4,'P': 7,'S':-1,'T':-1,'W':-4,'Y':-3,'V':-2,'X':0},
            'S':{'A': 1,'R':-1,'N': 1,'D': 0,'C':-1,'Q': 0,'E': 0,'G': 0,'H':-1,'I':-2,
                 'L':-2,'K': 0,'M':-1,'F':-2,'P':-1,'S': 4,'T': 1,'W':-3,'Y':-2,'V':-2,'X':0},
            'T':{'A': 0,'R':-1,'N': 0,'D':-1,'C':-1,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-1,
                 'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S': 1,'T': 5,'W':-2,'Y':-2,'V': 0,'X':0},
            'W':{'A':-3,'R':-3,'N':-4,'D':-4,'C':-2,'Q':-2,'E':-3,'G':-2,'H':-2,'I':-3,
                 'L':-2,'K':-3,'M':-1,'F': 1,'P':-4,'S':-3,'T':-2,'W':11,'Y': 2,'V':-3,'X':0},
            'Y':{'A':-2,'R':-2,'N':-2,'D':-3,'C':-2,'Q':-1,'E':-2,'G':-3,'H': 2,'I':-1,
                 'L':-1,'K':-2,'M':-1,'F': 3,'P':-3,'S':-2,'T':-2,'W': 2,'Y': 7,'V':-1,'X':0},
            'V':{'A': 0,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-2,'E':-2,'G':-3,'H':-3,'I': 3,
                 'L': 1,'K':-2,'M': 1,'F':-1,'P':-2,'S':-2,'T': 0,'W':-3,'Y':-1,'V': 4,'X':0},
            'X':{'A': 0,'R': 0,'N': 0,'D': 0,'C': 0,'Q': 0,'E': 0,'G': 0,'H': 0,'I': 0,
                 'L': 0,'K': 0,'M': 0,'F': 0,'P': 0,'S': 0,'T': 0,'W': 0,'Y': 0,'V': 0,'X':0}}

HYP1 = (255,   0,   0)
HYP2 = (128,   0,   0)
HYP3 = ( 64,   0,   0)
HYP4 = (  0,   0,   0)
HYP5 = (  0,  64, 128)
HYP6 = (  0, 128, 255)
HYP7 = ( 80,  80,  80)

COLOR_DICTS = {
               'aa':{'-':(192, 192, 192), 'A':(128, 128, 128), 'C':(255, 128,   0), 'B':(255, 128,   0),
                     'E':(255,   0,   0), 'D':(255,   0,   0), 'G':(128, 192,   0), 'F':( 40,  40,  40),
                     'I':( 92,  92,  92), 'H':(  0,  80, 160), 'K':(  0,  96, 255), 'J':(255, 128,   0),
                     'M':( 92,  92,  92), 'L':( 92,  92,  92), 'O':(255, 255, 128), 'N':(200,  40, 150),
                     'Q':(200,  40, 150), 'P':(  0, 180, 128), 'S':(192, 192,   0), 'R':(  0,  96, 255),
                     'T':(128, 128,   0), 'W':( 92,   0, 150), 'V':(128, 128, 128), 'Y':(128,   0, 128),
                     'X':(  0,   0,   0), 'Z':(255, 128,   0)},
               'ez':{'-':(192, 192, 192), 'A':(  0,  92,   0), 'C':(255, 128,   0), 'B':(255, 128,   0),
                     'E':(255,   0,   0), 'D':(255,   0,   0), 'G':( 92, 192,  92), 'F':(128,  64, 255),
                     'I':(128,   0, 128), 'H':(  0,  95, 255), 'K':(  0,  96, 255), 'J':(255, 128,   0),
                     'M':( 92,  92,  92), 'L':(  0,   0,   0), 'O':(255, 255, 128), 'N':(200, 200,   0),
                     'Q':(200, 200,   0), 'P':( 92, 192,  92), 'S':(200, 200,   0), 'R':(  0,  96, 255),
                     'T':(200, 200,   0), 'W':(128,  64, 255), 'V':(128, 128, 128), 'Y':(128,  64, 255),
                     'X':(  0,   0,   0), 'Z':(255, 128,   0)},
               'ch':
                    {'B':(255, 128, 128), 'C':(192, 128, 128),
                     'E':(255,   0,   0), 'D':(255,   0,   0),
                     'T':(160, 160,   0), 'S':(160, 160,   0), 'Y':(160, 160,   0), 'N':(160, 160,   0), 'Q':(160, 160,   0), 'W':(160, 160,   0), 
                     'H':( 64,  64, 192), 'K':(  0,  64, 255), 'R':(  0,  64, 255),
                     'Z':(255, 128, 128), '-':(192, 192, 192)},
                     
              'hyd':{'I':HYP1, 'A':HYP1, 'F':HYP1, 'L':HYP1, 'M':HYP1, 'P':HYP1, 'V':HYP1, 
                     'W':HYP2, 'C':HYP2, 'J':HYP2,
                     'G':HYP3, 'T':HYP3, 
                     'S':HYP4, 'Y':HYP4, 'O':HYP4,
                     'H':HYP5, 'Q':HYP5, 'N':HYP5, 
                     'K':HYP6, 'E':HYP6, 'D':HYP6, 'R':HYP6, 
                     'B':HYP7, 'Z':HYP7, 
                     'X':(0,0,0), '-':(192, 192, 192)},
                     
               'sz':{'-': (192, 192, 192), 'A': (206, 164, 27), 'C': (181, 145, 46), 'E': (116, 93, 98), 'D': (144, 115, 76), 
                     'G': (240, 192, 0), 'F': (54, 43, 148), 'I': (105, 84, 107), 'H': (104, 83, 108), 'K': (105, 84, 107),
                     'M': (100, 80, 111), 'L': (104, 83, 108), 'N': (139, 111, 80), 'Q': (114, 91, 100), 'P': (160, 128, 63),
                     'S': (191, 152, 39), 'R': (58, 46, 145), 'T': (157, 125, 66), 'W': (0, 0, 192), 'V': (139, 111, 80), 'Y': (38, 31, 160)}
                }             

CHAR_ADJ = {'A': (0.0, 1.370), 'C': (0.014, 1.320), 'E': (0.0, 1.370), 'D': (0.0, 1.370), 
            'G': (0.014, 1.320), 'F': (0.0, 1.370), 'I': (0.0, 1.370), 'H': (0.0, 1.370), 'K': (0.0, 1.370),
            'M': (0.0, 1.370), 'L': (0.0, 1.370), 'N': (0.0, 1.370), 'Q': (0.014, 1.320), 'P': (0.0, 1.370),
            'S': (0.014, 1.320), 'R': (0.0, 1.370), 'T': (0.0, 1.370), 'W': (0.0, 1.370), 'V': (0.0, 1.370), 'Y': (0.0, 1.370)}

class SvgDocument(object):

  def __init__(self, font='Arial'):

    self.font = font
    self._svg_lines = []

  def clear(self):

    self._svg_lines = []

  def svg(self, width, height):

    return ''.join(self._svg_head(width, height) + self._svg_lines + self._svg_tail())

  def write_file(self, file_name, width, height):

    if sys.version_info[0] < 3:
      import codecs
      open_func = codecs.open

    else:
      open_func = open

    with open_func(file_name, 'w', encoding='utf-8') as file_obj:
      file_obj.writelines(self._svg_head(width, height))
      file_obj.writelines(self._svg_lines)
      file_obj.writelines(self._svg_tail())
      file_obj.close()

  def _svg_head(self, width, height):

    head1 = '<?xml version="1.0"?>\n'
    head2 = '<svg height="%d" width="%d" image-rendering="optimizeSpeed" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.2" baseProfile="tiny">\n' % (height,width)

    return [head1, head2]

  def _svg_tail(self):

    return ['</svg>\n']
  
  def attrgroup_start(self, attrs):
    
    attrs = ' '.join('%s="%s"' % (k, v) for k, v in attrs.items())    
    
    self._svg_lines += '  <g %s>\n' % attrs
  
  def attrgroup_end(self):

    self._svg_lines.append('  </g>\n')  
    
    
  def group_start(self, opacity=1.0, stroke='black', stroke_opacity=1.0, width=1):

    data = (opacity, stroke_opacity, stroke, width)

    self._svg_lines += ['  <g style="fill-opacity:%.1f; stroke-opacity:%.1f; stroke:%s; stroke-width:%d;">\n' % data]

  def group_end(self):

    self._svg_lines.append('  </g>\n')

  def poly(self, coords, color='black', fill='blue', width=1):

    path = ['M']
    for x, y in coords:
      path += ['%d' % x, '%d' % y, 'L']
    path[-1] = 'Z'

    line = '    <path d="%s" fill="%s" stroke="%s" stroke-width="%d" />\n' % (' '.join(path), fill, color, width)
    self._svg_lines.append(line)

  def lines(self, coords, color='black', width=1):

    path = ['M']
    for x, y in coords:
      path += ['%d' % x, '%d' % y, 'L']
    del path[-1]

    line = '     <path d="%s" fill-opacity="0.0" stroke="%s" stroke-width="%.2fpx" />\n' % (' '.join(path), color, width)

    self._svg_lines.append(line)

  def line(self, coords, color='black', line_width=1):

    x1, y1, x2, y2 = coords
    line = '     <line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="%.2fpx" />\n' % (x1, y1, x2, y2, color, line_width)
    self._svg_lines.append(line)

  def circle(self, center, radius, color='black', fill='#808080'):

    cx, cy = center
    x = cx + radius
    y = cy + radius

    line = '     <circle cx="%d" cy="%d" r="%d" stroke="%s" fill="%s" />\n' % (x,y,radius,color,fill)
    self._svg_lines.append(line)

  def rect(self, coords, color='black', fill='#808080'):

    x0, y0, x1, y1 = coords

    x = min((x0, x1))
    y = min((y0, y1))
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    
    if fill:
      style = 'stroke:%s;fill:%s' % (color, fill)
    else:
      style = 'stroke:%s' % color  
    
    line = '     <rect x="%d" y="%d" height="%d" width="%d" style="%s"/>\n' % (x,y,h,w,style)
    
    self._svg_lines.append(line)

  def square(self, center, radius, color='black', fill='#808080'):

    cx, cy = center
    x = cx - radius
    y = cy - radius
    r = 2*radius
    line = '     <rect x="%d" y="%d" height="%d" width="%d" stroke="%s" fill="%s" />\n' % (x,y,r,r,color,fill)
    self._svg_lines.append(line)

  def text(self, text, coords, anchor='start', size=16, bold=False, font=True, color=None,
           angle=None, vert_align=None, sup=None, width=None, height=None, spacing=None, dy=None):

    if font:
      if font is True:
        font = ' font-family="%s"' % self.font
      else:
        font = ' font-family="%s"' % font
    else:
      font = ''
    
    if anchor:
      anchor = ' text-anchor="%s"' % anchor
    else:
      anchor = ''  

    x, y = coords

    attrs = ''
     
    if vert_align:
      attrs += ' dominant-baseline="%s"' % vert_align

    if color:
      attrs += ' fill="%s"' % color

    if angle:
      attrs += ' transform="rotate(%d %d %d)"' % (angle, x,y)

    if bold:
      attrs += ' font-weight="bold"'
    
    if spacing is not None:
      attrs += ' letter-spacing="%d" kerning="0"' % spacing
    
    if dy:
      attrs += ' dy="%s"' % dy
    
    if width:
      if height:
        size = height   
      
      attrs += ' textLength="%.2f" lengthAdjust="spacingAndGlyphs"' % width
      
    elif height:
      attrs += ' textLength="%.2f" lengthAdjust="spacingAndGlyphs"' % (size*len(text))
      size = height            
    
    if sup:
      sup = '<tspan font-size="%d" dy="%s">%s</tspan>\n' % ((2*size)/3,-size/2,sup)
    else:
      sup = ''

    text = saxutils.escape(text)
    line = '     <text x="%.2f" y="%.2f"%s%s font-size="%.2f"%s>%s%s</text>\n' % (x,y,anchor,font,size,attrs,text,sup)

    self._svg_lines.append(line)

  def segment(self, x0, y0, r, a1, a2, color='black', fill='grey', line_width=1):

    a1 = a1 % TAU
    a2 = a2 % TAU

    if a2 < a1:
      a1, a2 = a1, a2

    da = (a2-a1) % TAU
    la = 0 if da <= pi else 1
    clockwise = 1
    x1 = x0 + r * cos(a1)
    y1 = y0 + r * sin(a1)
    x2 = x0 + r * cos(a2)
    y2 = y0 + r * sin(a2)

    path = 'M %d %d L %d %d A %d %d %d %d %d %d %d Z' % (x0, y0, x1, y1, r, r, 0, la, clockwise, x2, y2)

    line = '    <path d="%s" fill="%s" stroke="%s" stroke-width="%d" />\n' % (path, fill, color, line_width)
    self._svg_lines.append(line)

  def image(self, x, y, w, h, data):

    io = bytes_io()
    img = misc.toimage(data)
    img.save(io, format="PNG")
    base_64_data = base64.b64encode(io.getvalue()).decode()

    line = '     <image x="%d" y="%d" width="%d" height="%d" xlink:href="data:image/png;base64,%s" />\n' % (x, y, w, h, base_64_data)

    self._svg_lines.append(line)

  def _graph_get_draw_coords(self, data_points, data_region, plot_region):

    dr0, dr1, dr2, dr3 = data_region
    pr0, pr1, pr2, pr3 = plot_region

    delta_x_plot = pr2 - pr0 or 1.0
    delta_y_plot = pr3 - pr1 or 1.0
    delta_x_data = dr2 - dr0 or 1.0
    delta_y_data = dr3 - dr1 or 1.0

    ppv_x = delta_x_plot/float(delta_x_data)
    ppv_y = delta_y_plot/float(delta_y_data)

    coords = []
    coords_append = coords.append
    for x, y, err in data_points:

      if (y is not None) and (x is not None):
        x0 = (x - dr0)*ppv_x
        y0 = (y - dr1)*ppv_y
        x0 += pr0
        y0 += pr1

        if err is None:
          e1 = e2 = None
        else:
          e0 = err * ppv_y
          e1 = y0 + e0
          e2 = y0 - e0

        if y0 < pr3:
          continue
        if y0 > pr1:
          continue

        coords_append((x0,y0,e1,e2))

    return coords

  def _graph_check_points(self, points):

    x_min = x_max = points[0][0]
    y_min = y_max = points[0][1]
    c = 0.0

    for i, point in enumerate(points):
      x, y = point[:2]

      x_min = min(x, x_min)
      y_min = min(y, y_min)
      x_max = max(x, x_max)
      y_max = max(y, y_max)

      if len(point) == 2:
        e = None
      else:
        e = point[2]

      if (not y) and (y != 0):
        y = None

      if (not x) and (x != 0):
        x = None

      else:
        if not isinstance(x, (float, int)):
          x = c
          c += 1.0

      points[i] = (x, y, e)

    return x_min, x_max, y_min, y_max

  def _hex_to_rgb(self, hex_code):

    r = int(hex_code[1:3], 16)
    g = int(hex_code[3:5], 16)
    b = int(hex_code[5:7], 16)

    return r, g, b

  def _default_color_func(self, matrix, pos_color, neg_color, bg_color):

    n, m = matrix.shape
    color_matrix = np.zeros((n, m, 3), float)
    rgb0 = np.array(self._hex_to_rgb(bg_color), float)
    rgbP = np.array(self._hex_to_rgb(pos_color), float)
    rgbN = np.array(self._hex_to_rgb(neg_color), float)

    for i in range(n):
      for j in range(m):
        f = matrix[i,j]

        if f > 0:
          g = 1.0 - f
          color_matrix[i,j] = (g * rgb0) + (f * rgbP)

        elif f < 0:
          f *= -1
          g = 1.0 - f
          color_matrix[i,j] = (g * rgb0) + (f * rgbN)
        else:
          color_matrix[i,j] = rgb0

    color_matrix = np.clip(color_matrix, 0, 255)
    color_matrix = np.array(color_matrix, dtype=np.uint8)

    return color_matrix

  def density_matrix(self, matrix, width, height=None, x_grid=None, y_grid=None,
                     x_labels=None, y_labels=None, x_axis_label=None, y_axis_label=None,
                     line_color='#000000', bg_color='#FFFFFF', grid_color='#808080',
                     pos_color='#0080FF', neg_color='#FF4000', color_func=None,
                     font=None, font_size=16, line_width=1, plot_offset=(50, 50),
                     value_range=None, scale_func=None, rotate_large_labels=True):

    pad = font_size + font_size / 4

    if not height:
      height = width

    if not font:
      font = self.font

    if not color_func:
      def color_func(x, p=pos_color, n=neg_color, b=bg_color):
        self._default_color_func(x, p, n, b)

    c_matrix = np.array(matrix)

    if value_range:
      a, b = value_range
      c_matrix = np.clip(c_matrix, a, b)

    if scale_func:
      c_matrix = scale_func(c_matrix)

    c_matrix /= max(c_matrix.max(), -c_matrix.min())

    if c_matrix.ndim == 2:
      n, m = c_matrix.shape
      d = 1
    else:
      n, m, d = c_matrix.shape

    x_box = width/float(m)
    y_box = height/float(n)

    x0, y0 = plot_offset

    x1, y1 = x0, y0

    if y_labels:
      x1 += pad

    if y_axis_label:
      x1 += pad

    x2 = x1 + width
    y2 = y1 + height

    c_matrix = color_func(c_matrix)

    self.rect((x1, y1, x2, y2), color=line_color, fill=bg_color)

    self.image(x1, y1, width, height, c_matrix)

    if x_grid:
      for val in x_grid:
        x = x1 + val * x_box
        self.line((x,y1,x,y2), color=grid_color, line_width=line_width)

    if y_grid:
      for val in y_grid:
        y = y1 + val * y_box
        self.line((x1,y,x2,y), color=grid_color, line_width=line_width)

    y3 = y2 + font_size/2
    x3 = x1 - font_size/2

    if x_labels:
      for i, val in enumerate(x_labels):
        if isinstance(val, (tuple, list)):
          x, t = val

        else:
          t = val
          x = i

        x = min(float(m), max(0.0, x))

        x = x1 + x_box/2.0 + x * x_box

        if rotate_large_labels:
          self.text(t, (x-font_size/4, y3), anchor='start', size=font_size-2, bold=False, font=font, color=line_color, angle=90, vert_align=None)

        else:
          self.text(t, (x, y3), anchor='middle', size=font_size-2, bold=False, font=font, color=line_color, angle=None, vert_align=None)

      y3 += pad

    if y_labels:
      for i, val in enumerate(y_labels):
        if isinstance(val, (tuple, list)):
          y, t = val

        else:
          t = val
          y = i

        y = min(float(n), max(0.0, y))
        y = y1 + y_box/2.0 + y * y_box

        if rotate_large_labels:
          self.text(t, (x3, y+font_size/4), anchor='end', size=font_size-2, bold=False, font=font, color=line_color, angle=None, vert_align=None)

        else:
          self.text(t, (x3, y), anchor='middle', size=font_size-2, bold=False, font=font, color=line_color, angle=270, vert_align=None)

      x3 -= pad

    if x_axis_label:
      x = m/2.0
      x = x1 + x * x_box

      y3 += pad
      self.text(x_axis_label, (x, y3), anchor='middle', size=font_size, bold=False, font=font, color=line_color, angle=None, vert_align=None)

    if y_axis_label:
      y = n/2.0
      y = y1 + y * y_box

      x3 -= pad
      self.text(y_axis_label, (x3, y), anchor='middle', size=font_size, bold=False, font=font, color=line_color, angle=270, vert_align=None)

  def graph(self, x, y, width, height, data_lists, x_label, y_label,
            names=None, colors=None,  graph_type=LINE_TYPE,
            symbols=None, line_widths=None, symbol_sizes=None,
            legend=False, title=None, x_labels=None, plot_offset=(100, 50),
            axis_color='black', bg_color='#F0F0F0', font=None, font_size=16, line_width=1,
            x_ticks=True, y_ticks=True, x_grid=False, y_grid=False,
            texts=None, opacities=None, x_log_base=None, y_log_base=None):

    n_data = len(data_lists)

    if not names:
      names = ['Data %d' % (i+1) for i in range(n_data)]

    if not font:
      font = self.font

    if not colors:
      colors = DEFAULT_COLORS

    if not symbols:
      symbols = ['circle' for i in range(n_data)]

    if not line_widths:
      line_widths = [1 for i in range(n_data)]

    if not symbol_sizes:
      symbol_sizes = [2 for i in range(n_data)]

    if not opacities:
      opacities = [None for i in range(n_data)]

    # Data region, check santity
    x_min, x_max, y_min, y_max = self._graph_check_points(data_lists[0])

    for data in data_lists[1:]:
      x_min1, x_max1, y_min1, y_max1 = self._graph_check_points(data)
      x_min = min(x_min, x_min1)
      y_min = min(y_min, y_min1)
      x_max = max(x_max, x_max1)
      y_max = max(y_max, y_max1)

    data_region = (x_min, y_min, x_max, y_max)

    # Inner plot region

    off_x, off_y, = plot_offset
    x0 = x + off_x
    y1 = y + off_y # Y increases down
    x1 = x0 + width - off_x
    y0 = y1 + height - off_y

    plot_region = (x0, y0, x1, y1)

    # Data region adjust

    if graph_type == BAR_TYPE:
      y_min = 0.0
      y_max *= 1.1
      x_min -= 0.5
      x_max += 0.5

    if x_min == x_max:
      x_max += 1.0

    if y_min == y_max:
      y_max += 1.0

    # Draw

    n_colors = len(colors)

    self.rect(plot_region, axis_color, fill=bg_color)
    self.line((x0,y0,x1,y0), axis_color, line_width)
    self.line((x0,y0,x0,y1), axis_color, line_width)

    self._graph_draw_ticks(data_region, plot_region, x_label, y_label,
                           line_width, x_ticks, y_ticks,
                           x_grid, y_grid, x_labels, x_log_base, y_log_base)

    if legend:
      if legend is True:
        x2 = x0 + (1.05*(x1-x0))
        y2 = y0 + (0.90*(y1-y0))
      else:
        x2, y2 = legend
        draw_coords = self._graph_get_draw_coords([(x2, y2, None)], data_region, plot_region)
        x2, y2 = draw_coords[0][:2]

      for i, data_set in enumerate(data_lists):
        name = names[i]

        if name:
          color = colors[i % n_colors]
          symbol = symbols[i] if graph_type in (LINE_TYPE, SCATTER_TYPE) else 'square'
          self._graph_draw_symbols([(x2, y2, None, None)], color, symbol, fill=color, symbol_size=7)
          self.text(name, (x2+font_size/2,y2), anchor='start', size=14, vert_align='middle')
          y2 += font_size

    for i, data_points in enumerate(data_lists):
      color = colors[i % n_colors]
      symbol = symbols[i]
      opacity = opacities[i]
      lw = line_widths[i]
      symbol_size = symbol_sizes[i]
      coords = self._graph_get_draw_coords(data_points, data_region, plot_region)

      if opacity is not None:
        self.group_start(stroke_opacity=opacity)

      if graph_type == LINE_TYPE:
        points = [point[:2] for point in coords]
        self.lines(points, color, lw)
        self._graph_draw_symbols(coords, color, symbol, symbol_size)

      elif graph_type == BAR_TYPE:
        self._graph_draw_boxes(coords, color, plot_region)
        self._graph_draw_symbols(coords, 'black', None, 8) # For error bars

      else: # SCATTER_TYPE
        self._graph_draw_symbols(coords, color, symbol, symbol_size, fill='#808080')

      if opacity is not None:
        self.group_end()

    if title:
      self.text(title, ((x0+x1)/2.0, y1/2.0), anchor='middle', size=font_size+2)

    if texts:
      for text, color, data_coords in texts:
        x, y = data_coords
        draw_coords = self._graph_get_draw_coords([(x, y, None)], data_region, plot_region)
        x, y = draw_coords[0][:2]
        self.text(text, (x+(font_size/2), y), color=color, vert_align='middle')

  def _graph_draw_boxes(self, coords, color, plot_region):

    points = [point[:2] for point in coords]
    y1 = plot_region[1]
    x_list = [p[0] for p in points]
    x_list.sort()

    deltas = [x_list[i+1]-x_list[i] for i in range(len(x_list)-1)]
    deltas.sort()
    width = deltas[0]/3.0

    for x, y in points:
      x1 = x - width
      x2 = x + width
      y2 = y
      self.rect((x1,y1,x2,y2), color='#000000', fill=color)

  def _graph_draw_symbols(self, coords, color, symbol, symbol_size, fill='#808080'):

    radius = symbol_size/2.0

    if symbol == 'square':
      for x, y, yU, yL in coords:
        x0 = x-radius
        x1 = x+radius
        y0 = y-radius

        if yU is not None:
          self.line((x,y,x,yU), color=color)
          self.line((x0,yU,x1,yU), color=color)

        if yL is not None:
          self.line((x,y,x,yL), color=color)
          self.line((x0,yL,x1,yL), color=color)

        self.square((x,y), radius-1, color=color, fill=color)

    elif symbol is None:
      for x, y, yU, yL in coords:
        x0 = x-radius
        x1 = x+radius
        y0 = y-radius

        if yU is not None:
          self.line((x,y,x,yU), color=color)
          self.line((x0,yU,x1,yU), color=color)

        if yL is not None:
          self.line((x,y,x,yL), color=color)
          self.line((x0,yL,x1,yL), color=color)

    else:
      for x, y, yU, yL in coords:
        x0 = x-radius
        x1 = x+radius
        y0 = y-radius

        if yU is not None:
          self.line((x,y,x,yU), color=color)
          self.line((x0,yU,x1,yU), color=color)

        if yL is not None:
          self.line((x,y,x,yL), color=color)
          self.line((x0,yL,x1,yL), color=color)

        self.circle((x0, y0), radius, color=color, fill=color)

  def _graph_draw_ticks(self, data_region, plot_region, x_label, y_label,
                        line_width=1, x_ticks=True, y_ticks=True,
                        x_grid=False, y_grid=False, x_labels=None,
                        x_log_base=None, y_log_base=None):

    def format_text(text):

      if isinstance(text, float):
        if text == 0:
          text = '0'
        elif abs(text) > 999999 or abs(text) < 0.01:
          text = '%5.2e' % text
        else:
          text = str(text)

      elif isinstance(text, int):
        text = str(text)

      if text and text[0:1] == '@':
        text = ''

      return text

    x0, y0, x1, y1 = plot_region
    delta_xplot = x1 - x0
    delta_yplot = y1 - y0
    delta_x_data = data_region[2] - data_region[0] or 1.0
    delta_y_data = data_region[3] - data_region[1] or 1.0

    y_close = 50
    x_close = 140

    xs = x0 - 5
    ys = y0 + 5

    xt = xs - 3
    yt = ys + 12

    ppv_x = delta_xplot/float(delta_x_data)
    ppv_y = delta_yplot/float(delta_y_data)

    space_x_data = x_close/ppv_x
    space_y_data = y_close/ppv_y

    sci_x = '%e' % abs(space_x_data)
    sci_y = '%e' % abs(space_y_data)

    deci_x = int(sci_x[-3:])
    deci_y = int(sci_y[-3:])

    sig_d_x = int(sci_x[0])
    sig_d_y = int(sci_y[0])

    n_x = 10.0
    n_y = 10.0
    s_x = abs(sig_d_x-n_x)
    s_y = abs(sig_d_y-n_y)
    for n in (1.0, 2.0, 5.0):
      s = abs(sig_d_x-n)
      if s < s_x:
        s_x = s
        n_x = n

      s = abs(sig_d_y-n)
      if s < s_y:
        s_y = s
        n_y = n

    inc_x = (abs(space_x_data)/space_x_data) *  n_x * 10**(deci_x) # noqa: E222
    inc_y = (abs(space_y_data)/space_y_data) * -n_y * 10**(deci_y)

    val_x = data_region[0] - (data_region[0] % inc_x)
    val_y = data_region[1] - (data_region[1] % inc_y)

    if x_ticks:
      for i in range(int(round(delta_x_data/inc_x))+2):

        tick_x = round(val_x,-deci_x)
        x = plot_region[0]+(tick_x - data_region[0])*ppv_x
        val_x += inc_x

        if x > plot_region[2]:
          continue
        if x < plot_region[0]:
          continue

        if x_labels and (i-1 < len(x_labels)):
          self.text(x_labels[i-1], (x, yt), anchor='middle')
        else:
          text = format_text(tick_x)
          if x_log_base:
            self.text(str(x_log_base), (x, yt), size=12, anchor='middle', sup=text)
          else:
            self.text(text, (x, yt), size=12, anchor='middle')

        if x_grid:
          self.line((x, y0, x, y1), 'black', 1)

        self.line((x, y0, x, ys), 'black', 1)

    if y_ticks:
      for i in range(int(round(delta_y_data/inc_y))+2):

        tick_y = round(val_y,-deci_y)
        y = plot_region[1]+(tick_y - data_region[1])*ppv_y
        val_y += inc_y

        if y < plot_region[3]:
          continue
        if y > plot_region[1]:
          continue

        text = format_text(tick_y)

        if y_log_base:
          self.text(str(y_log_base), (xt, y), size=12, anchor='end', vert_align='middle', sup=text)
        else:
          self.text(text, (xt, y), size=12, anchor='end', vert_align='middle')

        if y_grid:
          self.line((x0, y, x1, y), 'black', 1)

        self.line((x0, y, xs, y), 'black', 1)

    if x_label:
      self.text(x_label, ((x0+x1)/2.0, yt+16), size=14, anchor='middle')

    if y_label:
      self.text(y_label, (xt-30,((y0+y1)/2.0)), size=14, angle=-90, anchor='middle')

  def pie_chart(self, x, y, height, values, texts=None, colors=None, line_color='black', small_val=0, line_width=1, box_size=16, pad=4):

    rad = height/2 - 4
    x0 = x + height/2
    y0 = y + height/2

    n = float(sum(values))
    nv = len(values)

    if not colors:
      colors = [colorsys.hsv_to_rgb(i/(nv+1.0), 0.7, 0.8) for i in range(nv)]
      colors = ['#%02X%02X%02X' % (int(r*255), int(g*255), int(b*255)) for r, g, b in colors]

    nc = len(colors)
    a0 = -pi/2
    other = 0.0
    c = 0

    for i, value in enumerate(values):
      if value/n < small_val:
        other += value
        continue

      a1 = a0 + TAU * value/n
      self.segment(x0, y0, rad, a0, a1, line_color, colors[c%nc], line_width)
      a0 = a1
      c += 1

    if other:
      a1 = a0 + TAU * other/n
      self.segment(x0, y0, rad, a0, a1, line_color, colors[c%nc], line_width)

    x1 = x + height + 2*pad
    y1 = y + pad + box_size

    if texts:

      c = 0
      for i in range(nv):
        if values[i]/n < small_val:
          continue

        self.rect((x1, y1, x1+box_size, y1-box_size), color=line_color, fill=colors[c%nc])
        self.text(texts[i], (x1+box_size+pad, y1-2), anchor='start', size=box_size)

        y1 += box_size + pad
        c += 1

      if other:
        self.rect((x1, y1, x1+box_size, y1-box_size), color=line_color, fill=colors[c%nc])
        self.text('Other', (x1+box_size+pad, y1-2), anchor='start', size=box_size)

  def table(self, x0, y0, width, data, header=True, text_anchors=None, col_formats=None,
            size=16, pad=2, font=None, main_color='black', line_color='#808080', max_chars=64):

    row_height = size + pad + pad
    if not font:
      font = self.font

    if isinstance(width, float) and width < 1.0:
      width *= 1000

    n_cols = len(data[0])

    if not col_formats:
      col_formats = ['%s' for x in range(n_cols)]

    if not text_anchors:
      text_anchors = ['start' for x in range(n_cols)]

    col_widths = defaultdict(int)
    for row in data:
      for col, t in enumerate(row):
        if t is None:
          continue

        t = col_formats[col] % t
        col_widths[col] = min(max(len(t), col_widths[col]), max_chars)
    
    n = float(sum(col_widths.values()))

    col_widths = [width * col_widths[i]/n for i in range(n_cols)]

    x, y, = x0, y0

    self.line((x, y, x+width, y), color=line_color, line_width=1)

    for i, row in enumerate(data):
      y += row_height

      if (i == 0) and header:
        self.line((x, y, x+width, y), color=line_color, line_width=1)
        bold = True
      else:
        bold = False

      dx = 0
      for j, text in enumerate(row):
        if text is not None:
          text = col_formats[j] % text
          anchor = text_anchors[j]

          x1 = x + dx

          if anchor == 'start':
            x1 += pad

          elif anchor == 'middle':

            x1 += col_widths[j]/2

          else: # 'end'
            x1 += col_widths[j] - pad
          
          if len(text) > max_chars:
            frac = float(max_chars)/len(text)
            self.text(text, (x1, y-pad), anchor, size*frac, bold, font, color=main_color)
          
          else: 
            self.text(text, (x1, y-pad), anchor, size, bold, font, color=main_color)

        dx += col_widths[j]

    y += pad
    y += pad

    self.line((x, y, x+width, y), color=line_color, line_width=1)

    return width, y-y0


def read_fasta(stream_or_path, as_dict=True):
  
  if isinstance(stream_or_path, (str, unicode)):
    stream = open(stream_or_path)
  else:
    stream = stream_or_path
  
  named_seqs = []
  append = named_seqs.append
  name = None
  seq = []

  for line in stream:
    line = line.strip()
    
    if not line:
      continue
    
    if line[0] == '>':
      if name:
        append((name, ''.join(seq)))

      seq  = []
      name = line[1:]
    else:
      seq.append(line)

  if name:
    append((name, ''.join(seq)))

  if as_dict:
    return dict(named_seqs)
  else:
    return named_seqs
    
    
def logo_plot(seqs, color_mode='aa', col_width=14, plot_height=200,
              start_pos=None, end_pos=None, prior=0, pseudo_counts=1,
              seq_weights=None, font_size=14, pos_offset=0):
  
  """
  For BLOSUM see: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC146917/pdf/253389.pdf
  """
  
  svg_doc = SvgDocument()
  
  color_dict = COLOR_DICTS.get(color_mode, 'aa')
  
  n = len(seqs[0])
  m = len(seqs)
  
  if not start_pos:
    start_pos = 0
  
  if end_pos:
    end_pos = min(n, end_pos)
  else:
    end_pos = n
  
  if seq_weights is None:
    seq_weights = np.ones(m)
  else:
    seq_weights = np.array(seq_weights)
    seq_weights *= m/seq_weights.sum() # Preserve total count
  
  n = end_pos-start_pos
  
  x_off = font_size * 3.0
  y_off = font_size
  
  y_max = 0.01
  
  base = np.log2(20)
  aas = sorted(AMINO_ACID_PROP)
  items = []
  x = x_off
  x_max = x_off + n * col_width
  
  for i in range(start_pos, end_pos): # Each position
    freqs = defaultdict(int)
    
    ns = 0.0
    ng = 0.0
    for j in range(m): # Each sequence
      aa = seqs[j][i]
      w = seq_weights[j]
      
      if aa in aas:
        freqs[aa] += w
        ns += w   
      else:
        ng += w
    
    aminos = sorted(freqs)
    if ng:
      ns += 1.0 # ng/ns
      freqs['-'] = 1.0

    for aa in freqs:
      freqs[aa] /= ns
     
    mfreqs = {}
    for aa in aminos:
      mf = 0.0
 
      for aa2 in aminos:
        mf += freqs[aa2] * AMINO_ACID_PROP[aa] * exp(0.3176 * BLOSUM62[aa][aa2])
 
      mfreqs[aa] = mf
    
    w1 = float(m-1)
    for aa in mfreqs:
      w = (w1 * freqs[aa] + pseudo_counts * mfreqs[aa]) / (w1+pseudo_counts)
      freqs[aa] = w
        
    if prior == 1:
      prior_dict = AMINO_ACID_PROP
      
    elif prior == 2:
      prior_dict = defaultdict(float)
      t = 0.0
      
      for aa in freqs: # Given this residue
        f = freqs[aa]
      
        for aa2 in aas: # How often residues substitute
	  prob = f * AMINO_ACID_PROP[aa] * AMINO_ACID_PROP[aa2] * 2.0 ** BLOSUM62[aa][aa2] # frac_i * e_i * e_j * obs/exp
	  prior_dict[aa2] += prob
	  t += prob
     
      for aa in aas:
        prior_dict[aa] /= t 
      
    else:
      prior_dict = {}

    entropy = 0.0 
    for aa in freqs:
      h = freqs[aa] * np.log2(freqs[aa]/prior_dict.get(aa, 0.05))
      entropy += h
    
    heights = []
    for aa in aminos:
      heights.append((entropy *  freqs[aa], aa))

    heights.sort()     
    
    y = 0
    
    for dy, aa in heights:
      color='#%02X%02X%02X' % color_dict.get(aa, (128,128,128))
      items.append((x,y,aa,color,dy))
      
      y += dy
       
    x += col_width
    y_max = max(y_max, y)
  
  y_max *= 1.05
  y_scale = plot_height/float(y_max)
  svg_doc.rect((x_off,y_off,x_max,y_off+plot_height), fill='white')
  
  y_gap = 0.5
  attrs = {'text-anchor':"start",
           'font-family':'monospace'}
           
  svg_doc.attrgroup_start(attrs)
  
  for x, y, text, color, dy in items:
    height = y_scale*dy
    
    if height > 0.5:
      y = y_off+y_scale*(y_max-y)
      dy, stretch = CHAR_ADJ.get(text, (0.0, 1.0))
      h = height*stretch-y_gap
      
      if dy:
        dy = '-%.3f' % (dy * h)
      else:
        dy = None
        
      svg_doc.text(text, (x,y-y_gap), anchor=None, size=col_width, bold=True,
                   font=None, color=color, vert_align=None,
                   height=h, width=col_width, dy=dy)
    
      #svg_doc.rect((x,y,x+col_width,y-height), fill=None)
  
  svg_doc.attrgroup_end()
  
  # X ticks
  x_step = 5
  tick_len = min(3, font_size/2)
  for p in range(start_pos, end_pos+1):
    q = p + pos_offset
    
    if q % x_step == 0:
      i = p-start_pos
      x = x_off + i * col_width
      svg_doc.line( (x, y_off+plot_height, x, y_off+plot_height+tick_len) )
 
      if 0 < i < n:
        svg_doc.text('%d' % q, (x, y_off+plot_height+tick_len+font_size*0.8), size=font_size*0.8, anchor='middle')
 
  
  # Y ticks
  
  y_lim = int(math.ceil(y_max)) 
  y_step = y_lim/5.0
  
  if y_step < 1:
    y_step = int(5.0*y_step)/5.0
  else:
    y_step = int(y_step)

  for i in np.arange(0, y_max, y_step):
    y = y_off++ plot_height - (y_scale * i)
    svg_doc.line( (x_off-tick_len, y, x_off, y) )
    
    if y_step < 1:
      text = '%.1f' % i
    else:
      text = '%d' % i
    
    svg_doc.text(text, (x_off-1.5*tick_len, y), size=font_size*0.8, anchor='end', vert_align='central')
  
  svg_doc.text('Entropy (bits)', (font_size, y_off+plot_height/2.0), anchor='middle', angle=270)
  svg_doc.text('Alignment position', (x_off + 0.5 * (n * col_width), y_off + plot_height + 3.0 * font_size - tick_len), anchor='middle')
  
  
  #svg_doc.write_file('Test_chart.svg', x_max + 3 * (font_size), y_off + plot_height + tick_len + 3 * font_size)
    
  return svg_doc.svg(x_max + 3 * (font_size), y_off + plot_height + tick_len + 3 * font_size)
  
  
def logo_plot_fastas(fastas, start_pos=None, end_pos=None, prior=0, pseudo_counts=1, color_mode='ez',
                     suffix='_logo', number_offset=0, col_width=10, plot_height=140, font_size=14):

  for fasta_path in fastas:
 
    svg_file_path = os.path.splitext(fasta_path)[0] + suffix + '.svg'
    
    seqs = read_fasta(fasta_path).values()
    
    if start_pos is None:
      start_pos = 0
    
    if not end_pos:
      end_pos = max([len(s) for s in seqs])  

    svg = logo_plot(seqs, color_mode=color_mode, col_width=col_width, plot_height=plot_height,
                    start_pos=start_pos, end_pos=end_pos, prior=prior, pseudo_counts=pseudo_counts,
                    seq_weights=None, font_size=font_size, pos_offset=number_offset)
 
    with open(svg_file_path, 'w') as out_file_obj:
      out_file_obj.write(svg)

def main(argv=None):

  from argparse import ArgumentParser
  
  if argv is None:
    argv = sys.argv[1:]
  
  priors = ['flat', 'abundance', 'blosum']
  
  epilog = 'For further help email tjs23@cam.ac.uk'

  arg_parse = ArgumentParser(prog='logo_plot', description='Convert FASTA format alignhments to SVG format logo plot graphics',
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='FASTA_FILES', nargs='+', dest='i',
                         help='One or more protein sequence alignment file in FASTA format.')

  arg_parse.add_argument('-s', default=0, metavar='START_OFFSET', type=int,
                        help='Start position of the plot relative to the input alignment, defaults to first residue')  

  arg_parse.add_argument('-e', default=0, metavar='END_OFFSET', type=int,
                        help='End position of the plot relative to the input alignment, defaults to last residue')  
  
  arg_parse.add_argument('-o', metavar='OUTPUT_SUFFIX', default='_logo',
                         help='Output files will be named after their corresponding input FASTA file with the addition of the stated suffix plus ".svg". Default: "_logo"')

  arg_parse.add_argument('-p', metavar='PRIOR', default='flat',
                         help='Which prior to use when calculating entropy scores. Available: flat, abundance or blosum, Default: flat.')

  arg_parse.add_argument('-ps', default=1, metavar='PSEUDOCOUNT', type=int,
                         help='The pseudocount to add to help smooth aligments with few proteins. Default: 1')  


  args = vars(arg_parse.parse_args(argv))

  fastas = args['i']
  start_pos = args['s']
  end_pos = args['e']
  suffix = args['o']
  prior = args['p'].lower()
  pseudo = args['ps']
  
  if prior in priors:
    prior = priors.index(prior)
  else:
    print('Prior specification ust be one of "flat", "abundance" or "blosum"')
    sys.exit()
    
  if start_pos:
    start_pos -=1
  
  if end_pos:
    end_pos -= 1  
  
  logo_plot_fastas(fastas, start_pos=start_pos, end_pos=end_pos,
                   prior=prior, pseudo_counts=pseudo, color_mode='ez',
                   suffix=suffix, number_offset=0, col_width=10,
                   plot_height=140, font_size=14)

if __name__ == "__main__":
  sys.path.append(os.path.dirname(__file__))
  main()
