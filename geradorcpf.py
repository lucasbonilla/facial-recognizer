import random


def gerar():
    n = [random.randrange(10) for a in range(9)]
    o = 11 - sum(map(int.__mul__, n, range(10 + len(n) - 9, 1, -1))) % 11
    n += [(o >= 10 and [0] or [o])[0]]
    o = 11 - sum(map(int.__mul__, n, range(10 + len(n) - 9, 1, -1))) % 11
    n += [(o >= 10 and [0] or [o])[0]]
    return str("%d%d%d%d%d%d%d%d%d%d%d") % tuple(n)

if __name__ == '__main__':
    for i in range(100):
        cpf = gerar()
        print(cpf)