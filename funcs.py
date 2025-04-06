from random import randint as r
import math
mod1: int = 1
mod2: int = 2
maxD: float = 10


def act_n(a: float) -> float:
    if a != 0:
        return abs(a) / a * (abs(a) ** (mod1 / mod2))
    else:
        return 0


def act_v(v: list[float]) -> list[float]:
    result: list[float] = [0 for i in range(len(v))]
    for i in range(len(v)):
        if v[i] != 0:
            result[i] = abs(v[i]) / v[i] * abs(v[i]) ** (mod1 / mod2)
    return result


def dropping(n: int, prt: float) -> list[bool]:
    if prt != 0:
        result: list[bool] = []
        for i in range(n):
            s = r(0, int(1e7 // 1)) / 1e7
            if s < prt:
                result += [True]
            else:
                result += [False]
        if len(result) != n:
            raise KeyError
        return result
    else:
        return [False] * n


def normalize(v: list[float], t: int) -> list[list[float]]:
    result: list[float] = []
    for e in v:
        result += [math.exp(e / t)]
    R: list[float] = list(result)
    s = sum(result)
    for i in range(len(v)):
        result[i] /= s
    return [result, s, R]


def rationalize(v: list[float], t: float) -> list[float]:
    result: list[float] = []
    for e in v:
        result += [e // t * t]
    return result


def nv_mpx(n: float, v: list[float]) -> list[float]:
    result: list[float] = list(v)
    for i in range(len(v)):
        if v[i] != 0:
            result[i] = v[i] * n
    return result


def vv_sum(v1: list[float], v2: list[float], drops: list[bool] | None) -> list[float]:
    result: list[float] = [0 for i in range(len(v1))]
    if drops:
        for i in range(len(v1)):
            if not drops[i]:
                result[i] = v1[i] + v2[i]
    else:
        for i in range(len(v1)):
            result[i] = v1[i] + v2[i]
    return result


def mm_sum(m1: list[list[float]], m2: list[list[float]], drops: list[list[bool]] | None) -> list[list[float]]:
    result: list[list[float]] = [[0 for o in range(len(m1[i]))] for i in range(len(m1))]
    if drops:
        for i in range(len(m1)):
            for o in range(len(m1[i])):
                if not drops[i][o]:
                    result[i][o] = m1[i][o] + m2[i][o]
    else:
        for i in range(len(m1)):
            for o in range(len(m1[i])):
                result[i][o] = m1[i][o] + m2[i][o]
    return result


def nm_mpx(n: float, m: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = list(m)
    for i in range(len(m)):
        for o in range(len(m[i])):
            if m[i][o] != 0:
                result[i][o] = m[i][o] * n
    return result


def vm_mpx(v: list[float], m: list[list[float]], drops: list[bool] | None) -> list[float]:
    result: list[float] = [0 for i in range(len(m))]
    if drops:
        for i in range(len(drops)):
            if not drops[i]:
                for o in range(len(v)):
                    if v[o] != 0:
                        result[i] += v[o] * m[i][o]
    else:
        for i in range(len(m)):
            for o in range(len(v)):
                if v[o] != 0:
                    result[i] += v[o] * m[i][o]
    return result


def vm_mpx_wa(v: list[float], m: list[list[float]]) -> list[float]:
    result: list[float] = [0 for i in range(len(m))]
    for i in range(len(v)):
        if v[i] != 0:
            for o in range(len(m)):
                result[o] += m[o][i] * v[i]
    return result


def vv_mpx(v1: list[float], v2: list[float]) -> float:
    result: float = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result


def vm_div(do: list[float], m: list[list[float]], a: list[float]) -> list[float]:
    result: list[float] = [0 for i in range(len(a))]
    for i in range(len(a)):
        if a[i] != 0:
            for o in range(len(m)):
                if do[o] != 0:
                    result[i] += do[o] * m[o][i]
    for i in range(len(a)):
        if a[i] != 0:
            s = 1 / (abs(a[i]) ** (mod2 - mod1)) / mod2
            if s < maxD:
                result[i] *= s
            else:
                result[i] *= maxD
    return result


def vm_div_wa(do: list[float], m: list[list[float]]) -> list[float]:
    result: list[float] = [0 for i in range(len(m[0]))]
    for i in range(len(m[0])):
        for o in range(len(do)):
            result[i] += m[o][i] * do[o]
    return result


def vv_div(v1o: list[float], v2i: list[float]) -> list[list[float]]:
    result: list[list[float]] = [[0 for o in range(len(v2i))] for i in range(len(v1o))]
    for i in range(len(v1o)):
        if v1o[i] != 0:
            for o in range(len(v2i)):
                if v2i[o] != 0:
                    result[i][o] = v1o[i] * v2i[o]
    return result

