# Points on Line

2-dimensional points can be filtered based on their distances to given points quickly using a quadtree. Our aim is to quickly filter 2-dimensional points that are sufficiently close to a given line rather than a given point. This problem must be solved for many lines as efficiently as possible. In its simplest form a solution must implement the following function.

```python
Point = namedtuple('Point', ['x', 'y'])
Line = namedtuple('Line', ['a', 'b', 'c'])  # ax + by + c = 0 

def filter_points(points: list[Point], lines: list[Line], max_distance: float) -> list[set[int]]:

    #------ NAIVE SOLUTION BEGINS ------#

    filtered_points = []
    for line in lines:
        a, b, c = line
        filtered_points.append(set())
        for idx, point in enumerate(points):
            x, y = point
            distance = abs(a * x + b * y + c) / ((a ** 2 + b ** 2) ** 0.5)
            if distance <= max_distance:
                filtered_points[-1].add(idx)

    #------- NAIVE SOLUTION ENDS -------#

    assert len(filtered_points) == len(lines)  
    # filtered_points[i] is the indices of points that are close to lines[i].
    # (If and only if points[j] is close to lines[i], then j is an element of filtered_points[i].)
    return filtered_points
```
