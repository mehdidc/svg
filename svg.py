template = """
<svg version="1.0" width="300" height="300"
     xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink">
     {0}
</svg>
"""
path_template = """"<path d="M {x1},{y1} C {cx},{cy} {x2},{y2} {x3},{y3}"
                          stroke-width="5" style="fill:none;stroke:black;"/>"""


def gen_path(x1, y1, cx, cy, x2, y2, x3, y3):
    path = path_template.format(
        x1=x1, y1=y1,
        cx=cx, cy=cy,
        x2=x2, y2=y2,
        x3=x3, y3=y3)
    return path


def gen_svg_from_output(output):
    o = output.copy()
    o[:, (0, 2, 4, 6)] *= 200
    o[:, (0, 2, 4, 6)] += 50
    o[:, (1, 3, 5, 7)] *= 200
    o[:, (1, 3, 5, 7)] += 50
    assert output.shape[1] == 8
    s = ""
    for out in o:
        s += gen_path(*out) + "\n"
    return template.format(s)
