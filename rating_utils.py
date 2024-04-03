def calculate_precision_recall_f1_score(
    bitarray1: list[list[bool]], bitarray2: list[list[bool]]
) -> tuple[float, float, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(bitarray1)):
        for j in range(len(bitarray1[i])):
            if bitarray1[i][j] and bitarray2[i][j]:
                true_positive += 1
            elif bitarray1[i][j] and not bitarray2[i][j]:
                false_positive += 1
            elif not bitarray1[i][j] and bitarray2[i][j]:
                false_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score
