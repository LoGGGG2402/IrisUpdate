import sys


def shift(lst, n):
    return lst[-n:] + lst[:-n]


def xor(template1, template2):
    result = []
    for elem1, elem2 in zip(template1, template2):
        result.append(elem1 ^ elem2)
    return sum(result)


def hamming_distance(iris_code1, iris_code2):
    min_dist = sys.maxsize
    for i in range(-len(iris_code1[0]) // 2, len(iris_code1[0]) // 2 + 1):
        s = 0
        for j in range(len(iris_code1)):
            s += xor(iris_code1[j], shift(iris_code2[j], i))
        min_dist = min(min_dist, s)
    return min_dist / (len(iris_code1) * len(iris_code1[0]))


def matching(iris_code1, iris_code2, threshold=0.32):
    res = hamming_distance(iris_code1, iris_code2)
    print(res)
    if res <= threshold:
        return True, 1 - res
    return False, 1 - res
