import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import argparse
import logging
import os

# Sample command:
# python linear_regression.py --csv_path ../data/lm_updated_substituted_edge_accuracy_po.csv --groupby --use_pairwise_sim

def load_csv(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path, sep='\t')
    return df

def setup_logger(args):
    # Create logs directory if it doesn't exist
    os.makedirs("../logs", exist_ok=True)
    # Build log file name based on input arguments
    model_name = args.csv_path.split('/')[-1].split('_')[0]
    model_type = args.csv_path.split('_')[-5]
    model_output_type = args.csv_path.split('_')[-1].split('.')[0]
    sim_measure = 'pairwise_sim' if args.use_pairwise_sim else 'mean_sim'
    groupby = 'groupby' if args.groupby else 'aggregated'
    log_filename = f"../logs/{model_name}_{model_type}_{model_output_type}_{sim_measure}_{groupby}.log"
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        level=logging.INFO,
        # format='%(asctime)s %(levelname)s: %(message)s'
        format='%(message)s'
    )
    return logging.getLogger()

# loop through sim csv file
# need a container to store x, y values for later regression
def aggregated_results(df, logger):
    acc_y = []
    sim_mean_x1 = []
    sim_pairwise_x2 = []
    for index, row in df.iterrows():
        # get the concept1 and concept2
        concept1 = row['concept1']
        concept2 = row['concept2']
        # get the row in the accuracy csv file when concept 1= concept 1 and concept2 = concept 2 and the value in the accuracy column
        accuracy_score = row['accuracy']
        # print the concept out in one line
        logger.info(f"concept1: {concept1}, concept2: {concept2}, accuracy: {accuracy_score}")
        similarity_mean = row['similarity_Mean']
        similarity_pairwise = row['similarity_pairwise']
        # append the values to the x, y, z lists
        acc_y.append(accuracy_score)
        sim_mean_x1.append(similarity_mean)
        sim_pairwise_x2.append(similarity_pairwise)
    return acc_y, sim_mean_x1, sim_pairwise_x2

def group_results(df):
    #dictionary to store the values
    grouped_data = {}
    grouped = df.groupby('category')

    for name, group in grouped:
        acc_y = group['accuracy'].tolist()
        sim_mean_x1 = group['similarity_Mean'].tolist()
        sim_pairwise_x2 = group['similarity_pairwise'].tolist()

        grouped_data[name] = {
            'acc_y': acc_y,
            'mean_sim': sim_mean_x1,
            'pairwise_sim': sim_pairwise_x2
        }

    return grouped_data

def main(args):
    logger = setup_logger(args)
    df = load_csv(args.csv_path)
    sim_measure = 'pairwise_sim' if args.use_pairwise_sim else 'mean_sim'
    logger.info(f"sim_measure: {sim_measure}")
    model_output_type = args.csv_path.split('_')[-1].split('.')[0]
    model_name = args.csv_path.split('/')[-1].split('_')[0]
    model_type = args.csv_path.split('_')[-5]

    if args.groupby:
        grouped_data = group_results(df)
        group_r2 = {}

        for category, data in grouped_data.items():
            acc_y = np.array(data['acc_y'])
            if args.use_pairwise_sim:
                sim_x = np.array(data['pairwise_sim']).reshape(-1, 1)
            else:
                sim_x = np.array(data['mean_sim']).reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(sim_x, acc_y)
            y_pred = model.predict(sim_x)
            r2 = r2_score(acc_y, y_pred)
            group_r2[category] = r2

            logger.info(f"\nCategory: {category}")
            logger.info(f"  Intercept: {model.intercept_}")
            logger.info(f"  Slope: {model.coef_[0]}")
            logger.info(f"  R² score: {r2:.4f}")
            
            # plot the data and the regression line
            plt.figure()
            plt.scatter(sim_x, acc_y, color="blue", label="Data")
            plt.plot(sim_x, y_pred, color="red", label="Regression line")
            plt.xlabel(sim_measure)
            plt.ylabel("Accuracy")
            plt.title(f"{category}: {model_name} {model_type} {model_output_type}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"../plots/category/{category}_{model_name}_{model_type}_{model_output_type}_{sim_measure}.png")
            plt.close()
        
        # remove nan r2
        group_r2 = {k: v for k, v in group_r2.items() if not np.isnan(v)}

        # Rank and save to file
        logger.info("\n=== Group Ranking by R² Score ===")
        ranking_file = f"../data/{model_name}_{model_type}_{model_output_type}_{sim_measure}_ranking.txt"
        with open(ranking_file, 'w') as f:
            f.write("=== Group Ranking by R² Score ===\n")
            for i, (cat, r2) in enumerate(sorted(group_r2.items(), key=lambda x: x[1], reverse=True), 1):
                line = f"{i}. {cat}: {r2:.4f}"
                logger.info(line)
                f.write(line + '\n')

    else:
        acc_y, sim_mean, sim_pairwise = aggregated_results(df, logger)
        if args.use_pairwise_sim:
            x1 = np.array(sim_pairwise).reshape(-1, 1)
        else:
            x1 = np.array(sim_mean).reshape(-1, 1)
        y = np.array(acc_y)
        # print("x1 shape:", x1.shape)
        # print("y shape:", y.shape)
        
        model = LinearRegression()
        model.fit(x1, y)
        logger.info(f"Intercept: {model.intercept_}")
        logger.info(f"Slope: {model.coef_[0]}")
        
        y_pred = model.predict(x1)
        logger.info(f"R² score: {r2_score(y, y_pred)}")
        
        # plot the data and the regression line
        plt.scatter(x1, y, color="blue", label="Data")
        plt.plot(x1, y_pred, color="red", label="Regression line")
        plt.xlabel(sim_measure)
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} {model_type} {model_output_type}")
        plt.legend()
        plt.show()
        plt.savefig(f"../plots/aggregated/{model_name}_{model_type}_{model_output_type}_{sim_measure}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression on similarity/accuracy data.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the input CSV file.") # Resulting substituted edge accuracy csv files from compute_taxonomy_sims_image.py
    parser.add_argument('--groupby', action='store_true', help="Whether to group by category.")
    parser.add_argument('--use_pairwise_sim', action='store_true', help="Uses pairwise similarity if true, otherwise uses mean similarity.")
    args = parser.parse_args()
    main(args)