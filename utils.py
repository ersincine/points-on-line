import random

import cv2 as cv
import numpy as np
from constants import Line, Point


def visualize(width: int, height: int, points: list[Point], lines: list[Line], filtered_points: list[set[int]]) -> None:
    img = 255 * np.ones((height, width, 3), np.uint8)

    # Show all points
    for point in points:
        x, y = point
        cv.circle(img, (round(x), round(y)), 2, (0, 0, 0), -1)

    # Show lines and points that are close to the lines.
    for line, points_on_line in zip(lines, filtered_points):
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        a, b, c = line
        cv.line(img, (0, int(-c / b)), (width, int((-c - a * width) / b)), random_color, 2)
        for idx in points_on_line:
            x, y = points[idx]
            cv.circle(img, (round(x), round(y)), 2, random_color, -1)

    cv.imshow("img", img)
    cv.waitKey(0)


def calculate_line_that_passes_through_two_points(point1: Point, point2: Point) -> Line:
    a = point2[1] - point1[1]
    b = point1[0] - point2[0]
    c = -a * point1[0] - b * point1[1]
    assert abs(a * point2[0] + b * point2[1] + c) < 1e-6
    return Line(a, b, c)


def generate_random_point(min_x: float, max_x: float, min_y: float, max_y: float) -> Point:
    return Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))


def generate_random_line(width: int, height: int, certainly_visible: bool=False) -> Line:
    if certainly_visible:
        point1 = generate_random_point(0, width, 0, height)
        point2 = generate_random_point(0, width, 0, height)
    else:
        # Generate two points that are not necessarily visible.
        # Division by 3 is an arbitrary choice.
        point1 = generate_random_point(-width//3, width + width//3, -height//3, height + height//3)
        point2 = generate_random_point(-width//3, width + width//3, -height//3, height + height//3)

    return calculate_line_that_passes_through_two_points(point1, point2)


def is_line_visible(width: int, height: int, line: Line) -> bool:
    a, b, c = line
    if a == 0:
        return 0 <= -c / b <= height
    if b == 0:
        return 0 <= -c / a <= width
    return 0 <= -c / b <= height or 0 <= (-c - a * width) / b <= height or 0 <= -c / a <= width or 0 <= (-c - b * height) / a <= width

