import numpy as np
import matplotlib.pyplot as plt

def parseInput():
    numbers = []
    with open("data.txt", "r") as f:
        data = f.read().splitlines()
    for i in range(len(data)):
        tmp = data[i].split(",")
        numbers.append([int(tmp[0]), int(tmp[1])])
    return np.array(numbers, np.int32)


if __name__ == "__main__":
    data = parseInput()
