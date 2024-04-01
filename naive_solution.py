from constants import Line, Point
from utils import calculate_distance_between_point_and_line

def filter_points(points: list[Point], lines: list[Line], max_distance: float) -> list[set[int]]:

    filtered_points = []
    for line in lines:
        filtered_points.append(set())
        for idx, point in enumerate(points):
            distance = calculate_distance_between_point_and_line(point, line)
            if distance <= max_distance:
                filtered_points[-1].add(idx)

    assert len(filtered_points) == len(lines)  
    # filtered_points[i] is the indices of points that are close to lines[i].
    # (If and only if points[j] is close to lines[i], then j is an element of filtered_points[i].)
    return filtered_points
