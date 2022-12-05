import math


def euclidean(_a: list, _b: list):
    return math.dist(_a, _b)


def manhattan(_a: list, _b: list):
    return sum([abs(i - j) for i, j in zip(_a, _b)])


def minkowski(_a: list, _b: list, _exp=2):
    return sum([abs(i - j) ** _exp for i, j in zip(_a, _b)]) ** (1/_exp)
