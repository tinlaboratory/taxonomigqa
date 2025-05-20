import config
import csv
import json
import pathlib
import random

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon, norm, ttest_rel

# from statsmodels.stats.weightstats import ttest_ind
from tqdm import tqdm


def get_utri(matrix):
    triu_indices = np.triu_indices_from(matrix, k=1)

    # Extract the upper triangular elements
    matrix_flat = (1 - matrix)[triu_indices]
    return matrix_flat


def compute_rsa(matrix1, matrix2, similarity_metric="spearman"):
    # Get the upper triangular indices (excluding diagonal)
    triu_indices = np.triu_indices_from(matrix1, k=1)

    # Extract the upper triangular elements
    matrix1_flat = (1 - matrix1)[triu_indices]
    matrix2_flat = (1 - matrix2)[triu_indices]

    # Compute correlation between the flattened similarity matrices
    if similarity_metric == "pearson":
        similarity, p_value = pearsonr(matrix1_flat, matrix2_flat)
    elif similarity_metric == "spearman":
        similarity, p_value = spearmanr(matrix1_flat, matrix2_flat)

    return similarity, p_value


def save_matrix(matrix, path):
    data = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            data.append((i, j, matrix[i, j]))

    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "sim"])
        writer.writerows(data)


def save_matrices(mats, dir_path):

    dist, lda, mean = mats
    save_matrix(dist, f"{dir_path}/dist.csv")
    save_matrix(lda, f"{dir_path}/lda.csv")
    save_matrix(mean, f"{dir_path}/mean.csv")


def load_matrices(model_name, save=True):
    path = f"reps/{model_name}/"
    dist, original_lda, original_lda_shuff = np.load(f"{path}/lda.npy")
    dist, original_mean, original_mean_shuff = np.load(f"{path}/mean.npy")

    if save:
        save_path = f"reps/{model_name}/long-mats/"
        pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)

        save_matrices((dist, original_lda, original_mean), save_path)

    # return dist, original_lda, original_lda_shuff, original_mean, original_mean_shuff
    return {
        "dist": dist,
        "original_lda": original_lda,
        "original_lda_shuff": original_lda_shuff,
        "original_mean": original_mean,
        "original_mean_shuff": original_mean_shuff,
    }


def compute_rsa_diff(matrix1, matrix2, reference, num_samples=100, num_trials=100):
    """non parametric test of differences between"""

    assert matrix1.shape == matrix2.shape == reference.shape

    rsas_1_ref = []
    rsas_2_ref = []

    for i in range(num_trials):
        matrix1_subset = np.zeros((num_samples, num_samples))
        matrix2_subset = np.zeros((num_samples, num_samples))
        reference_subset = np.zeros((num_samples, num_samples))

        random_rows = random.sample(range(matrix1.shape[0]), num_samples)

        # print(random_rows)

        # populate
        for i in range(len(random_rows)):
            for j in range(len(random_rows)):
                matrix1_subset[i, j] = matrix1[random_rows[i], random_rows[j]]
                matrix2_subset[i, j] = matrix2[random_rows[i], random_rows[j]]
                reference_subset[i, j] = reference[random_rows[i], random_rows[j]]

        # compute rsa between each matrix and ref
        rsa_1_ref = compute_rsa(reference_subset, matrix1_subset)
        rsa_2_ref = compute_rsa(reference_subset, matrix2_subset)
        rsas_1_ref.append(rsa_1_ref[0])
        rsas_2_ref.append(rsa_2_ref[0])

    # non parameteric testing
    diff = np.array(rsas_1_ref) - np.array(rsas_2_ref)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    test = wilcoxon(diff, method="asymptotic")

    eff_size = abs(norm.ppf(test.pvalue / 2)) / np.sqrt(len(rsas_1_ref))

    return diff_mean, test.pvalue, eff_size


def compute_diff_rsa_pair(matrix1, matrix2, num_samples=100, num_trials=100):
    """non parametric test of differences between"""

    assert matrix1.shape == matrix2.shape

    rsas = []
    m1_sims = []
    m2_sims = []
    # m_diffs = []

    for i in range(num_trials):
        matrix1_subset = np.zeros((num_samples, num_samples))
        matrix2_subset = np.zeros((num_samples, num_samples))
        # reference_subset = np.zeros((num_samples, num_samples))

        random_rows = random.sample(range(matrix1.shape[0]), num_samples)

        # print(random_rows)

        # populate
        for i in range(len(random_rows)):
            for j in range(len(random_rows)):
                matrix1_subset[i, j] = matrix1[random_rows[i], random_rows[j]]
                matrix2_subset[i, j] = matrix2[random_rows[i], random_rows[j]]
                # reference_subset[i, j] = reference[random_rows[i], random_rows[j]]

        # compute rsa between each matrix and ref
        # rsa_1_ref = compute_rsa(reference_subset, matrix1_subset)
        # rsa_2_ref = compute_rsa(reference_subset, matrix2_subset)
        rsa = compute_rsa(matrix1_subset, matrix2_subset)
        rsas.append(rsa[0])
        m1_sim = get_utri(matrix1_subset)
        m2_sim = get_utri(matrix2_subset)
        # diff = np.mean(m1_sim - m2_sim)
        # m_diffs.append(diff)
        m1_sims.append(np.mean(m1_sim))
        m2_sims.append(np.mean(m2_sim))
        # rsas_1_ref.append(rsa_1_ref[0])
        # rsas_2_ref.append(rsa_2_ref[0])

    # non parameteric testing
    mean_rsa = np.mean(rsas)
    sd_rsa = np.std(rsas)
    diff = np.array(m1_sims) - np.array(m2_sims)
    # diff = m_diffs
    diff_mean = np.mean(diff)
    diff_std = np.std(diff, ddof=1)
    eff_size = diff_mean / diff_std

    test = ttest_rel(m1_sims, m2_sims)
    # test = wilcoxon(diff, method="asymptotic")

    # eff_size = abs(norm.ppf(test.pvalue/2))/np.sqrt(len(rsas))

    return diff_mean, test.pvalue, eff_size, mean_rsa, sd_rsa


def compute_cross_metrics(vision_model, text_model):
    """
    computes:

    v vs. wn lda
    v-s vs. wn lda
    l vs. wn lda
    l-s vs. wn lda
    v vs. l lda
    v vs. wn mean
    v-s vs. wn mean
    l vs. wn mean
    l-s vs. wn mean
    v vs. l mean

    where v/l = vision vs. language only
    X-s = shuffled version of X
    wn = wordnet based inverse distance matrix
    """

    vision_mats = load_matrices(vision_model)
    text_mats = load_matrices(text_model)

    vl_lda_results = compute_rsa_diff(
        vision_mats["original_lda"], text_mats["original_lda"], text_mats["dist"]
    )
    vl_mean_results = compute_rsa_diff(
        vision_mats["original_mean"], text_mats["original_mean"], text_mats["dist"]
    )

    vl_lm_lda_diff = compute_diff_rsa_pair(
        vision_mats["original_lda"], text_mats["original_lda"]
    )

    vl_lm_lda_diff_results = (vl_lm_lda_diff[0], vl_lm_lda_diff[1], vl_lm_lda_diff[2])
    vlm_lm_lda_rsa_mean = (vl_lm_lda_diff[3], 0, 0)
    vlm_lm_lda_rsa_sd = (vl_lm_lda_diff[4], 0, 0)

    vl_wn_lda_diff = compute_diff_rsa_pair(
        vision_mats["original_lda"], vision_mats["dist"]
    )

    vlm_wn_lda_rsa_mean = (vl_wn_lda_diff[3], 0, 0)
    vlm_wn_lda_rsa_sd = (vl_wn_lda_diff[4], 0, 0)

    lm_wn_lda_diff = compute_diff_rsa_pair(text_mats["original_lda"], text_mats["dist"])

    lm_wn_lda_rsa_mean = (lm_wn_lda_diff[3], 0, 0)
    lm_wn_lda_rsa_sd = (lm_wn_lda_diff[4], 0, 0)

    return {
        "v_wn_lda": compute_rsa(vision_mats["dist"], vision_mats["original_lda"]),
        "v_s_wn_lda": compute_rsa(
            vision_mats["dist"], vision_mats["original_lda_shuff"]
        ),
        "v_wn_mean": compute_rsa(vision_mats["dist"], vision_mats["original_mean"]),
        "v_s_wn_mean": compute_rsa(
            vision_mats["dist"], vision_mats["original_mean_shuff"]
        ),
        "l_wn_lda": compute_rsa(text_mats["dist"], text_mats["original_lda"]),
        "l_s_wn_lda": compute_rsa(text_mats["dist"], text_mats["original_lda_shuff"]),
        "l_wn_mean": compute_rsa(text_mats["dist"], text_mats["original_mean"]),
        "l_s_wn_mean": compute_rsa(text_mats["dist"], text_mats["original_mean_shuff"]),
        "v_l_lda": compute_rsa(vision_mats["original_lda"], text_mats["original_lda"]),
        "v_l_mean": compute_rsa(
            vision_mats["original_mean"], text_mats["original_mean"]
        ),
        "v_l_non_param_lda_diff": vl_lda_results,
        "v_l_non_param_diff": vl_mean_results,
        "v_l_non_param_diff_sims": vl_lm_lda_diff_results,
        "v_l_sampled_rsa_mean": vlm_lm_lda_rsa_mean,
        "v_l_sampled_rsa_sd": vlm_lm_lda_rsa_sd,
        "v_wn_lda_rsa_mean": vlm_wn_lda_rsa_mean,
        "v_wn_lda_rsa_sd": vlm_wn_lda_rsa_sd,
        "l_wn_lda_rsa_mean": lm_wn_lda_rsa_mean,
        "l_wn_lda_rsa_sd": lm_wn_lda_rsa_sd,
    }


metric_results = []

for pair in tqdm(config.PAIRS):
    vision = pair["vision"].replace("/", "_")
    text = pair["text"].replace("/", "_")

    cross_metrics = compute_cross_metrics(vision, text)

    for k, v in cross_metrics.items():
        if "non_param" in k:
            metric_results.append((pair["class"], k, v[0], v[1], v[2]))
        else:
            metric_results.append((pair["class"], k, v[0], v[1], 0))

with open("data/results/pair-rsa.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(("model_class", "metric", "score", "pvalue", "eff_size"))
    writer.writerows(metric_results)
