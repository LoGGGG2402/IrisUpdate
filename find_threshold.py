import os

import matplotlib.pyplot as plt

from feature_extraction import encode_iris
from matching import matching, hamming_distance


def read_file():
    """Reads the file and returns a list of the lines in the file."""
    root = 'database'
    folders = [folder for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]
    files = []

    for folder in folders:
        left = os.path.join(root, folder, 'left')
        right = os.path.join(root, folder, 'right')
        left_files = [os.path.join(left, file) for file in os.listdir(left) if file.endswith('.bmp')]
        right_files = [os.path.join(right, file) for file in os.listdir(right) if file.endswith('.bmp')]
        for file in left_files:
            name = folder + "left"
            files.append((name, file))
        for file in right_files:
            name = folder + "right"
            files.append((name, file))
    print('Read file map successfully.')
    return files


def get_iris_code():
    """Returns a list of tuples of the form (name, iris_code)"""
    files = read_file()
    iris_codes = []
    for file in files:
        code = encode_iris(file[1])
        if code is not None:
            iris_codes.append((file[0], code))
            print('\r', (len(iris_codes) / len(files) * 100).__str__() + '% of codes are extracted.', end='')

    # write the iris code to a file
    with open('iris_code.txt', 'w') as f:
        for i in iris_codes:
            f.write(i[0] + ": " + str(i[1]) + "\n")
    return iris_codes


def read_iris_code():
    """Reads the iris code from the file."""
    iris_code = []
    with open('iris_code.txt', 'r') as f:
        for line in f:
            line = line.split(": ")
            name = line[0]
            code = line[1]
            code = code[2:-3]
            code = code.split('], [')
            code = [[int(i) for i in j.split(', ')] for j in code]
            iris_code.append((name, code))
    with open('iris_code1.txt', 'w') as f:
        for i in iris_code:
            f.write(i[0] + ": " + str(i[1]) + "\n")
    return iris_code


def find_hamming_distance(iris_code):
    distance = []
    for i in range(len(iris_code)-1):
        row = []
        for j in range(len(iris_code)-1):
            match1 = (iris_code[i][0] == iris_code[j][0])
            dis1 = hamming_distance(iris_code[i][1], iris_code[j][1])
            row.append((match1, dis1))
        distance.append(row)
    return distance


def security_level(distance, threshold):
    """Returns the security level of the threshold."""
    acceptance = 0
    rejection = 0
    false_acceptance = 0
    false_rejection = 0
    for i in range(len(distance) - 1):
        for j in range(i + 1, len(iris_code)):
            match, confidence = matching(iris_code[i][1], iris_code[j][1], threshold)
            if iris_code[i][0] == iris_code[j][0]:
                if match:
                    acceptance += 1
                else:
                    false_rejection += 1
            else:
                if match:
                    false_acceptance += 1
                else:
                    rejection += 1
    far = false_acceptance / (false_acceptance + acceptance)
    frr = false_rejection / (false_rejection + rejection)
    return far, frr





def find_threshold(model='get', from_x=1, to_x=1000):
    """Returns the threshold that gives the best security level."""
    if model == 'read':
        iris_code = read_iris_code()
    else:
        iris_code = get_iris_code()
    far = []
    frr = []
    # threshold = 0.001 - 1
    threshold = [i / to_x for i in range(from_x, to_x)]
    with open('threshold.txt', 'w') as f:
        with open('far_frr.txt', 'w') as g:
            for i in threshold:
                far_i, frr_i = security_level(iris_code, i)
                far.append(far_i)
                frr.append(frr_i)
                f.write(str(i) + "\n")
                g.write(str(far_i) + " " + str(frr_i) + "\n")
    min_far = min(far)
    min_frr = min(frr)
    index_far = far.index(min_far)
    index_frr = frr.index(min_frr)
    print("min_far: ", min_far, "index_far: ", index_far)
    print("min_frr: ", min_frr, "index_frr: ", index_frr)
    print("threshold for min_far: ", threshold[index_far])
    print("threshold for min_frr: ", threshold[index_frr])
    return threshold, far, frr


def write_result():
    """Draws the graph of far and frr."""
    threshold, far, frr = find_threshold()
    # write the result to a file
    with open('result.txt', 'w') as f:
        f.write("threshold: " + str(threshold) + "\n")
        f.write("far: " + str(far) + "\n")
        f.write("frr: " + str(frr) + "\n")
    plt.plot(threshold, far, label='far')
    plt.plot(threshold, frr, label='frr')
    plt.xlabel('threshold')
    plt.ylabel('far/frr')
    plt.legend()
    plt.show()


def read_result():
    """Reads the result from the file."""
    with open('result.txt', 'r') as f:
        threshold = f.readline()
        threshold = threshold[12:-2]
        threshold = threshold.split(', ')
        threshold = [float(i) for i in threshold]
        far = f.readline()
        far = far[6:-2]
        far = far.split(', ')
        far = [float(i) for i in far]
        frr = f.readline()
        frr = frr[6:-2]
        frr = frr.split(', ')
        frr = [float(i) for i in frr]
    return threshold, far, frr


def draw_far_frr_from_file():
    """Draws the graph of far and frr from the file."""
    threshold, far, frr = read_result()
    plt.plot(threshold, far, label='far')
    plt.plot(threshold, frr, label='frr')
    plt.xlabel('threshold')
    plt.ylabel('far/frr')
    plt.legend()
    plt.show()


def main():
    write_result()
    # draw_far_frr_from_file()


if __name__ == '__main__':
    main()
