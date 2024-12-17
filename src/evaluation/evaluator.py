import pandas as pd
import argparse

def loose_answer_match(ground_truth: str, model_output: str) -> bool:
    return ground_truth.lower() in model_output.lower()

def strict_answer_match(ground_truth: str, model_output: str) -> bool:
    selected_output = model_output.lower().split()[0]
    return ground_truth.lower() in selected_output

def cal_accuracy(correct: int, total: int, q_type: str) -> float:
    acc = (correct / total * 100) if total != 0 else 0
    print(f"The {q_type} accuracy is: {acc:.2f}% ({correct}/{total})")
    return acc

def process_accuracy(df: pd.DataFrame, eval_type: str) -> list:
    # Calculate type counts
    type_counts = {
        i: len(df[df['question_type'] == i]) 
        for i in range(5)  # 0 through 4
    }
    
    results = []
    for q_type, desc in enumerate(['overall', 'original question'] + [f'{i}-level question' for i in ['first', 'second', 'third', 'fourth']]):
        if q_type == 0:
            correct = len(df[df[eval_type]])
            total = len(df)
        else:
            correct = len(df[(df['question_type'] == q_type-1) & (df[eval_type])])
            total = type_counts[q_type-1]
        results.append(cal_accuracy(correct, total, desc))
    
    print("\n")
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate model outputs against ground truth')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the data')
    parser.add_argument('--eval_type', type=str, default='loose_eval',
                       choices=['loose_eval', 'strict_eval'],
                       help='Evaluation type: loose_eval or strict_eval')
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.data_path, sep='\t')
    
    # Vectorized operations instead of loop
    df['loose_eval'] = df.apply(lambda row: loose_answer_match(str(row['ground_truth']), str(row['model_output'])), axis=1)
    df['strict_eval'] = df.apply(lambda row: strict_answer_match(str(row['ground_truth']), str(row['model_output'])), axis=1)
    
    # Process overall results
    overall_result = process_accuracy(df, args.eval_type)
    
    # Initialize dictionary with list comprehension
    dict_output = {f"type_{i}": [] for i in range(5)}
    dict_output["overall"] = []
    
    # Process by group
    for _, group in df.groupby('image_id'):
        group_res = process_accuracy(group, args.eval_type)
        for i, key in enumerate(['overall'] + [f'type_{j}' for j in range(5)]):
            dict_output[key].append(group_res[i])
    
    # std_list = [statistics.stdev(val) for val in dict_output.values()]
    # print(std_list)

if __name__ == "__main__":
    main()
    