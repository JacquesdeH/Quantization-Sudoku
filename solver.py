# Program to solve Sudokus using backtracking algorithm
# All possible solutions are found

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from multiprocessing import Pool


def solve(bo, solutions):
    find = find_empty(bo)
    if not find:
        # FOUND SOLUTION
        tmp = copy.deepcopy(bo)
        solutions.append(tmp)
        return
    else:
        row, col = find
    for i in range(1, 10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            solve(bo, solutions)
            bo[row][col] = 0


def valid(bo, num, pos):
    # Check the row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check the columns 
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check the square
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if bo[i][j] == num and (i, j) != pos:
                return False
    return True  


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                # row, col
                return (i, j)
    return None


def run_solver(bo: list) -> list:
    # import pdb
    # pdb.set_trace()
    answers = []
    solve(bo, answers)
    return answers


def str2list(line: str) -> list:
    ret = [[0 for _ in range(9)] for _ in range(9)]
    for x in range(9):
        for y in range(9):
            idx = x * 9 + y
            digit = int(line[idx])
            ret[x][y] = digit
    return ret


def list2str(grid: list) -> str:
    ret = [None] * 81
    for x in range(9):
        for y in range(9):
            idx = x * 9 + y
            ret[idx] = str(grid[x][y])
    return ''.join(ret)


def main(sudoku_file: str, output_file: str, L: int, R: int):
    _multi_quiz_cnt = 0
    sudoku_df = pd.read_csv(sudoku_file)
    output_df = pd.DataFrame(columns=['quizzes', 'solutions'])
    for _index in tqdm(sudoku_df.index):
        if not (L <= _index <= R):
            # L and R is closed interval
            continue
        # if _index == 100:
        #     import pdb
        #     pdb.set_trace()
        _quiz_line, _solution_gt = sudoku_df.loc[_index]
        _quiz = str2list(_quiz_line)
        _answers = run_solver(_quiz)
        assert len(_answers) >= 1
        if len(_answers) > 1:
            _multi_quiz_cnt += 1
            # import pdb
            # pdb.set_trace()
        # append to output df
        for _answer in _answers:
            _answer_line = list2str(_answer)
            output_df = output_df.append({'quizzes': _quiz_line, 'solutions': _answer_line}, ignore_index=True)
    output_df.to_csv(output_file, index=False)
    return _multi_quiz_cnt


def main_wrapper(args):
    return main(*args)


if __name__ == '__main__':
    # board = [
    #     [0,0,0,0,0,0,0,0,3],
    #     [0,5,6,9,0,0,0,0,0],
    #     [0,9,0,0,3,4,0,0,0],
    #     [0,6,0,0,8,9,5,0,0],
    #     [0,0,4,6,0,2,8,0,0],
    #     [0,0,8,4,5,0,0,1,0],
    #     [0,0,0,3,9,0,0,7,0],
    #     [0,0,0,0,0,1,2,5,0],
    #     [8,0,0,0,0,0,0,0,0]
    # ]
    # answers = run_solver(board)
    
    # sudoku_file = '/mnt/lustre/hezexin/workspace/Projects/quantum-sudoku-large-homework/data/sudoku.csv'
    # sudoku_df = pd.read_csv(sudoku_file)
    # _quiz, _solution_gt = sudoku_df.loc[0]

    # _line = _quiz
    # _grid = str2list(_line)
    # _line_hat = list2str(_grid)
    # import pdb
    # pdb.set_trace()
    total_entry = 1000000
    process_cnt = 100
    delta = total_entry // process_cnt

    intervals = [(idx*delta, idx*delta+delta-1) for idx in range(process_cnt)]

    parallel_args = [
        ('./data/sudoku.csv', f'./data/sudoku_extended_{lower}_{upper}.csv', lower, upper) \
        for (lower, upper) in intervals
    ]

    with Pool(processes=process_cnt) as pool:
        results = pool.map(main_wrapper, parallel_args)

    print(f'Multi quiz count @ {results}')
