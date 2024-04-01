from turtle import width
from constants import Line, Point
import naive_solution
from utils import calculate_line_that_passes_through_two_points, generate_random_line, generate_random_point, is_line_visible, visualize


def generate_random_problem(
        width: int=1000, 
        height: int=500, 
        point_count: int=1000, 
        line_count: int=40, 
        distance: float=20.0,
        visible_lines_only: bool=False) \
        -> tuple[list[Point], list[Line], float]:
    
    points = [generate_random_point(0, width, 0, height) for _ in range(point_count)]
    lines = [generate_random_line(width, height, visible_lines_only) for _ in range(line_count)]
    #ratio_visible_lines = sum(is_line_visible(width, height, line) for line in lines) / line_count
    #print(f"ratio_visible_lines: {ratio_visible_lines}")
    return points, lines, distance


def generate_random_problem_with_vanishing_point(
        width: int=1000, 
        height: int=500, 
        point_count: int=1000, 
        line_count: int=40, 
        distance: float=20.0,
        vanishing_point_certainly_visible: bool=False) \
        -> tuple[list[Point], list[Line], float]:
    
    points = [generate_random_point(0, width, 0, height) for _ in range(point_count)]

    lines = []
    if vanishing_point_certainly_visible:
        vanishing_point = generate_random_point(0, width, 0, height)
    else:
        # Generate a point that is not necessarily visible.
        # Division by 2 is an arbitrary choice.
        vanishing_point = generate_random_point(-width//3, width + width//3, -height//3, height + height//3)
    for _ in range(line_count):
        # Generate another point very close to the vanishing point.
        other_point = generate_random_point(vanishing_point[0] - 1, vanishing_point[0] + 1, vanishing_point[1] - 1, vanishing_point[1] + 1)
        line = calculate_line_that_passes_through_two_points(vanishing_point, other_point)
        lines.append(line)

    #ratio_visible_lines = sum(is_line_visible(width, height, line) for line in lines) / line_count
    #print(f"ratio_visible_lines: {ratio_visible_lines}")
    return points, lines, distance


def main() -> None:
    width, height = 1000, 500
    distance = 20
    line_count = 20

    points, lines, distance = generate_random_problem(width, height, distance=distance, line_count=line_count)
    filtered_points = naive_solution.filter_points(points, lines, distance)
    visualize(width, height, points, lines, filtered_points, distance)

    points, lines, distance = generate_random_problem_with_vanishing_point(width, height, distance=distance, line_count=line_count)
    filtered_points = naive_solution.filter_points(points, lines, distance)
    visualize(width, height, points, lines, filtered_points, distance)


if __name__ == "__main__":
    main()
