import time
import torch

import naive_solution
from constants import Line, Point
from main import generate_random_problem
from rating_utils import calculate_precision_recall_f1_score


def filter_points(
    points: list[Point],
    lines: list[Line],
    max_distance: float,
    device=torch.device("cpu"),
) -> tuple[list[set[int]], float]:
    start = time.time()
    with torch.no_grad():
        points_tensor = torch.tensor(
            [[point.x, point.y, 1] for point in points],
            device=device,
        )
        lines_tensor = torch.tensor(
            [[line.a, line.b, line.c] for line in lines],
            device=device,
        )

        distances = torch.abs(torch.matmul(lines_tensor, points_tensor.T)) / torch.norm(
            lines_tensor[:, :2], dim=1, keepdim=True
        )

        end = time.time()
        return (
            (distances <= max_distance).detach().cpu().numpy(),
            end - start,
        )


def to_bitarray(filtered_points: list[set[int]], point_count: int) -> list[list[bool]]:
    return [
        [
            True if point_idx in filtered_points[line_idx] else False
            for point_idx in range(point_count)
        ]
        for line_idx in range(len(filtered_points))
    ]


def main():
    naive_times = []
    pytorch_cpu_times = []
    pytorch_gpu_times = []

    precision_scores_cpu = []
    recall_scores_cpu = []
    precision_scores_gpu = []
    recall_scores_gpu = []

    for _ in range(6):
        print(f"Test {_ + 1}...")
        width, height = 1000, 1000
        distance = 20
        point_count = 4000
        line_count = 4000

        points, lines, distance = generate_random_problem(
            width,
            height,
            distance=distance,
            point_count=point_count,
            line_count=line_count,
        )

        start = time.time()
        filtered_points = naive_solution.filter_points(points, lines, distance)
        naive_times.append(time.time() - start)
        naive_solution_bitarray = to_bitarray(filtered_points, len(points))

        pytorch_cpu_solution_bitarray, pytorch_cpu_time = filter_points(
            points, lines, distance, device=torch.device("cpu")
        )
        pytorch_cpu_times.append(pytorch_cpu_time)

        assert torch.cuda.is_available()
        pytorch_qpu_solution_bitarray, pytorch_gpu_time = filter_points(
            points, lines, distance, device=torch.device("cuda")
        )
        pytorch_gpu_times.append(pytorch_gpu_time)

        precision_cpu, recall_cpu, _ = calculate_precision_recall_f1_score(
            naive_solution_bitarray, pytorch_cpu_solution_bitarray
        )
        precision_scores_cpu.append(precision_cpu)
        recall_scores_cpu.append(recall_cpu)

        precision_gpu, recall_gpu, _ = calculate_precision_recall_f1_score(
            naive_solution_bitarray, pytorch_qpu_solution_bitarray
        )
        precision_scores_gpu.append(precision_gpu)
        recall_scores_gpu.append(recall_gpu)

    print(
        "Naive solution average time:",
        sum(naive_times) / len(naive_times),
    )
    print(
        "PyTorch CPU solution average time:",
        sum(pytorch_cpu_times) / len(pytorch_cpu_times),
    )
    print(
        "PyTorch GPU solution average time:",
        sum(pytorch_gpu_times) / len(pytorch_gpu_times),
    )

    precision_cpu_average = sum(precision_scores_cpu) / len(precision_scores_cpu)
    recall_cpu_average = sum(recall_scores_cpu) / len(recall_scores_cpu)
    precision_gpu_average = sum(precision_scores_gpu) / len(precision_scores_gpu)
    recall_gpu_average = sum(recall_scores_gpu) / len(recall_scores_gpu)

    print(f"{precision_cpu_average=}")
    print(f"{recall_cpu_average=}")
    print(
        "PyTorch CPU f1 score average:",
        2
        * precision_cpu_average
        * recall_cpu_average
        / (precision_cpu_average + recall_cpu_average),
    )

    print(f"{precision_gpu_average=}")
    print(f"{recall_gpu_average=}")
    print(
        "PyTorch GPU f1 score average:",
        2
        * precision_gpu_average
        * recall_gpu_average
        / (precision_gpu_average + recall_gpu_average),
    )


if __name__ == "__main__":
    main()
