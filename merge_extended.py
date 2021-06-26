import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    total_entry = 1000000
    process_cnt = 100
    delta = total_entry // process_cnt

    intervals = [(idx*delta, idx*delta+delta-1) for idx in range(process_cnt)]

    partials = [f'./data/sudoku_extended_{lower}_{upper}.csv'\
        for (lower, upper) in intervals]
    
    output_df = pd.DataFrame(columns=['quizzes', 'solutions'])

    for _file in partials:
        _df = pd.read_csv(_file)
        output_df = output_df.append(_df)
    
    output_df.to_csv('./data/sudoku_extended.csv', index=False)

    print('Successfully merged !')
