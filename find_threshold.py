import concurrent.futures
import os
import pickle

import matplotlib.pyplot as plt

from feature_extraction import encode_iris
from matching import hamming_distance


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


def calculate_distance(args):
    i, j, iris_code = args
    match1 = (iris_code[i][0] == iris_code[j][0])
    dis1 = hamming_distance(iris_code[i][1], iris_code[j][1])
    return i, j, match1, dis1


def find_hamming_distance(iris_code):
    distances = [[None] * len(iris_code) for _ in range(len(iris_code))]

    total_tasks = len(iris_code) * (len(iris_code) - 1) // 2  # Total number of tasks
    max_workers = os.cpu_count() or 1
    print(max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = []
        for i in range(len(iris_code)):
            for j in range(i + 1, len(iris_code)):
                results.append(executor.submit(calculate_distance, (i, j, iris_code)))

        for result in concurrent.futures.as_completed(results):
            percentage = (i / total_tasks * 100)
            print(f"\rProgress: {percentage:.2f}%   " + result.result().__str__(), end='', flush=True)
            results.append(result.result())

    # Update distance array with results
    for result in results:
        i, j, match, dis = result
        distances[i][j] = (match, dis)
        distances[j][i] = (match, dis)

    print('\nHamming distance')
    with open('distance.pkl', 'wb') as file:
        pickle.dump(distances, file)

    return distances


def security_level(args):
    distances, threshold = args
    """Returns the security level of the threshold."""
    acceptance = 0
    rejection = 0
    false_acceptance = 0
    false_rejection = 0
    for i in range(len(distances) - 1):
        for j in range(i + 1, len(distances[i])):
            h_distance = distances[i][j]
            match = (h_distance[1] < threshold)
            true_match = h_distance[0]
            if true_match:
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
    return far, frr, threshold


def find_threshold(model='read', from_x=1, to_x=1000):
    """Returns the threshold that gives the best security level."""
    if model == 'read':
        iris_code = read_iris_code()
    else:
        iris_code = get_iris_code()

    distances = find_hamming_distance(iris_code)
    far = []
    frr = []
    # threshold = 0.001 - 1
    threshold = [i / to_x for i in range(from_x, to_x)]
    with open('far_frr.txt', 'w') as g:
        max_workers = os.cpu_count() or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = []
            for i in threshold:
                results.append(executor.submit(security_level, (distances, i)))

            for result in concurrent.futures.as_completed(results):
                far_i, frr_i, threshold = result.result()
                far.append(far_i)
                frr.append(frr_i)
                g.write(str(far_i) + " " + str(frr_i) + " " + str(threshold) + "\n")
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
