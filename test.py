import numpy as np

def main():
    A = np.array([[1, 0, 0.5, 0.5],[0, 1, 0.5, 0.5], [0.5, 0.5, 1, 0], [0.5, 0.5, 0, 1]])
    temp = np.linalg.inv(A)
    print(temp)


if __name__ == "__main__":
    main()
