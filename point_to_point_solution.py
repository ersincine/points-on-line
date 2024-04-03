import time
import torch
import numpy as np

from scipy.spatial import KDTree
from constants import Point
from rating_utils import calculate_precision_recall_f1_score
from utils import generate_random_point


def naive_filter_points_mask(
    points: list[Point], max_distance: float
) -> tuple[list[list[bool]], float]:

    def calculate_distance_between_points(point1: Point, point2: Point) -> float:
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

    filtered_points = []
    for point1 in points:
        filtered_points.append([])
        for point2 in points:
            distance = calculate_distance_between_points(point1, point2)
            filtered_points[-1].append(distance <= max_distance)

    assert len(filtered_points) == len(points)
    return filtered_points


def quick_filter_points_mask(
    points: list[Point],
    max_distance: float,
    device=torch.device("cpu"),
) -> tuple[list[set[int]], float]:
    start = time.time()
    with torch.no_grad():
        points_tensor = torch.tensor(
            [[point.x, point.y] for point in points],
            device=device,
        )

        distances = torch.cdist(points_tensor, points_tensor)

        end = time.time()
        return (
            (distances <= max_distance).detach().cpu().numpy(),
            end - start,
        )


def kdtree_filter_points_mask(
    points: list[Point],
    max_distance: float,
) -> tuple[list[list[bool]], float]:
    start = time.time()

    points_numpy = np.array([[point.x, point.y] for point in points])

    kdtree = KDTree(points_numpy)
    distances = kdtree.query_ball_point(points_numpy, r=max_distance)
    end = time.time()

    distances_bitarray = [[False] * len(points) for _ in range(len(points))]
    for point1_idx in range(len(points)):
        for point2_idx in distances[point1_idx]:
            distances_bitarray[point1_idx][point2_idx] = True

    return (
        distances_bitarray,
        end - start,
    )


def evaluate(
    datasets: list[list[Point]],
    ground_truths: list[list[list[bool]]],
    test_count: int,
    function: callable,
    **kwargs,
) -> tuple[float, float, float, float]:
    runtimes = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for test_iter in range(test_count):
        result, runtime = function(datasets[test_iter], **kwargs)
        runtimes.append(runtime)

        precision, recall, f1 = calculate_precision_recall_f1_score(
            ground_truths[test_iter], result
        )
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return (
        sum(runtimes) / test_count,
        sum(precision_scores) / test_count,
        sum(recall_scores) / test_count,
        sum(f1_scores) / test_count,
    )


def main():
    test_count = 6
    width, height = 1000, 1000
    distance = 20
    point_count = 4000

    naive_times = []
    datasets = []
    ground_truths = []
    for i in range(test_count):
        datasets.append(
            [generate_random_point(0, width, 0, height) for _ in range(point_count)]
        )

        start = time.time()
        naive_solution_bitarray = naive_filter_points_mask(datasets[i], distance)
        naive_times.append(time.time() - start)
        ground_truths.append(naive_solution_bitarray)

    (
        pytorch_cpu_average_runtime,
        pytorch_cpu_average_precision,
        pytorch_cpu_average_recall,
        pytorch_cpu_average_f1,
    ) = evaluate(
        datasets,
        ground_truths,
        test_count,
        quick_filter_points_mask,
        max_distance=distance,
        device=torch.device("cpu"),
    )

    (
        pytorch_gpu_average_runtime,
        pytorch_gpu_average_precision,
        pytorch_gpu_average_recall,
        pytorch_gpu_average_f1,
    ) = evaluate(
        datasets,
        ground_truths,
        test_count,
        quick_filter_points_mask,
        max_distance=distance,
        device=torch.device("cuda"),
    )

    (
        kdtree_average_runtime,
        kdtree_average_precision,
        kdtree_average_recall,
        kdtree_average_f1,
    ) = evaluate(
        datasets,
        ground_truths,
        test_count,
        kdtree_filter_points_mask,
        max_distance=distance,
    )

    print(f"{test_count} tests were run.")
    print(f"Naive solution average time: {sum(naive_times) / test_count}")
    print(f"PyTorch CPU solution average time: {pytorch_cpu_average_runtime}")
    print(f"PyTorch GPU solution average time: {pytorch_gpu_average_runtime}")
    print(f"KDTree solution average time: {kdtree_average_runtime}")
    print()
    print(f"PyTorch CPU solution average precision: {pytorch_cpu_average_precision}")
    print(f"PyTorch CPU solution average recall: {pytorch_cpu_average_recall}")
    print(f"PyTorch CPU solution average f1: {pytorch_cpu_average_f1}")
    print()
    print(f"PyTorch GPU solution average precision: {pytorch_gpu_average_precision}")
    print(f"PyTorch GPU solution average recall: {pytorch_gpu_average_recall}")
    print(f"PyTorch GPU solution average f1: {pytorch_gpu_average_f1}")
    print()
    print(f"KDTree solution average precision: {kdtree_average_precision}")
    print(f"KDTree solution average recall: {kdtree_average_recall}")
    print(f"KDTree solution average f1: {kdtree_average_f1}")
    print()


if __name__ == "__main__":
    main()
