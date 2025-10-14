import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import spacy
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import f_oneway, kruskal, mannwhitneyu, shapiro, levene, t
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
from tqdm import tqdm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# Define paths
DATA_PATH = "Narrative_Analysis_output/text_segmentation/to_the_lighthouse_paragraphs.csv"
OUTPUT_DIR = "Narrative_Analysis_output/stylistic_feature"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pbar = tqdm(total=8, desc="Global Progress", position=0)

# Read data
df = pd.read_csv(DATA_PATH)
df.columns = ["part", "chapter_id", "para_idx", "text"]

part_order = ['The Window', 'Time Passes', 'The Lighthouse']
df['part_num'] = df['part'].map({p: i for i, p in enumerate(part_order)})

pbar.update(1)

# Load NLP model
nlp = spacy.load("en_core_web_sm", disable=["ner"])
docs = [doc for doc in nlp.pipe(df["text"].tolist(), batch_size=32)]
pbar.update(1)

# Function to compute tree depth
def get_tree_depth(root):
    return 1 + max((get_tree_depth(c) for c in root.children), default=0)

# Function to extract stylistic features
def calculate_stylistic_features(doc):
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    num_words = len([t for t in doc if t.is_alpha])
    
    is_single_sentence = 1 if num_sentences == 1 else 0
    
    num_adj_adv = len([t for t in doc if t.pos_ in ['ADJ','ADV']])
    num_verbs = len([t for t in doc if t.pos_ == 'VERB'])
    pos_ratio_adj_adv_to_verb = num_adj_adv / max(num_verbs, 1)
    
    dep_depths = [get_tree_depth(sent.root) / max(len([w for w in sent if w.is_alpha]), 1) for sent in sentences]
    avg_dependency_depth = sum(dep_depths) / max(len(dep_depths), 1)

    num_subclauses = len([t for t in doc if t.dep_ in {"advcl","ccomp","xcomp","csubj","acl","relcl"}])
    subclauses_per_sentence = num_subclauses / max(num_sentences, 1)
    
    content_words = [t.text.lower() for t in doc if t.pos_ in ['NOUN','VERB','ADJ','ADV'] and t.is_alpha]
    repeated_words = [w for w, count in pd.Series(content_words).value_counts().items() if count > 1]
    repeated_words_per_sentence = len(repeated_words) / max(num_sentences, 1)
    
    parallel_conjs = len([t for t in doc if t.dep_ == 'cc' and t.head.pos_ == 'VERB'])
    parallel_ratio = parallel_conjs / max(num_sentences, 1)
    
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(doc.text)
    compound_sentiment = sentiment['compound']
    
    return pos_ratio_adj_adv_to_verb, avg_dependency_depth, subclauses_per_sentence, repeated_words_per_sentence, parallel_ratio, compound_sentiment, is_single_sentence

features = [calculate_stylistic_features(doc) for doc in docs]
feat_df = pd.DataFrame(features, columns=[
    'pos_ratio_adj_adv_to_verb', 'avg_dependency_depth', 'subclauses_per_sentence', 
    'repeated_words_per_sentence', 'parallel_ratio', 'compound_sentiment', 'is_single_sentence'
])
df = pd.concat([df, feat_df], axis=1)

df = df.sort_values(['part_num', 'chapter_id', 'para_idx']).reset_index(drop=True)
df = df.drop('part_num', axis=1)
pbar.update(1)

# Generate embeddings and reduce dimensions
model = SentenceTransformer("all-MiniLM-L12-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=False)
pca_coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
df['pca_x'], df['pca_y'] = pca_coords[:,0], pca_coords[:,1]
umap_coords = umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=2, random_state=42).fit_transform(embeddings)
df['umap_x'], df['umap_y'] = umap_coords[:,0], umap_coords[:,1]
pbar.update(1)

# Function for Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    diff = x.mean() - y.mean()
    pooled_std = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return diff / pooled_std if pooled_std != 0 else 0  # Avoid division by zero

# Function for 95% CI
def calculate_ci(data, alpha=0.05):
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    se = sd / np.sqrt(n)
    t_val = t.ppf(1 - alpha/2, n-1)
    ci_lower = mean - t_val * se
    ci_upper = mean + t_val * se
    return ci_lower, ci_upper

# Statistical tests
metrics = ['pos_ratio_adj_adv_to_verb', 'avg_dependency_depth', 'subclauses_per_sentence', 
           'repeated_words_per_sentence', 'parallel_ratio', 'compound_sentiment']
results = []
assumption_tests = []
for metric in metrics:
    groups = [df[df['part']==p][metric].dropna().values for p in part_order]
    fval, pval = f_oneway(*groups)
    results.append((metric, 'ANOVA', fval, pval, np.nan))
    hval, pval = kruskal(*groups)
    results.append((metric, 'Kruskal', hval, pval, np.nan))
    for i1, i2 in combinations(range(len(part_order)), 2):
        p1, p2 = part_order[i1], part_order[i2]
        data1 = df[df['part']==p1][metric].dropna()
        data2 = df[df['part']==p2][metric].dropna()
        u_stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
        eff_size = cohens_d(data1, data2)
        results.append((metric, f'Mann-Whitney {p1} vs {p2}', u_stat, p_val, eff_size))
    
    # Assumption tests
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        levene_stat, levene_p = levene(*groups)
        assumption_tests.append({
            'Metric': metric,
            'Test': 'Levene (Homogeneity)',
            'Statistic': levene_stat,
            'p-value': levene_p
        })
    for i, group_data in enumerate(groups):
        if len(group_data) >= 3:
            shapiro_stat, shapiro_p = shapiro(group_data)
            assumption_tests.append({
                'Metric': metric,
                'Test': f'Shapiro-Wilk ({part_order[i]})',
                'Statistic': shapiro_stat,
                'p-value': shapiro_p
            })

results_df = pd.DataFrame(results, columns=["feature", "test", "stat", "pval", "eff_size"])
results_df["pval_corrected"] = multipletests(results_df["pval"], method="fdr_bh")[1]
pbar.update(1)

assumption_tests_df = pd.DataFrame(assumption_tests).round(4)

# Compute descriptive statistics
key_metrics = ['pos_ratio_adj_adv_to_verb', 'avg_dependency_depth', 'compound_sentiment', 'is_single_sentence']
desc_combined = []
for metric in key_metrics:
    for part in part_order:
        data = df[df['part'] == part][metric].dropna()
        desc = {
            'Feature': metric,
            'Part': part,
            'Count': len(data),
            'Mean': data.mean(),
            'Std': data.std() if metric != 'is_single_sentence' else np.nan,
            'Median': data.median() if metric != 'is_single_sentence' else np.nan
        }
        if metric == 'is_single_sentence':
            desc['Mean'] = desc['Mean']  # Proportion for is_single_sentence
            desc['Std'] = np.nan
            desc['Median'] = np.nan
        else:
            ci_lower, ci_upper = calculate_ci(data)
            desc['95% CI Lower'] = ci_lower
            desc['95% CI Upper'] = ci_upper
        desc_combined.append(desc)

desc_combined_df = pd.DataFrame(desc_combined).round(4)
pbar.update(1)

# Visualization
sns.set(style="whitegrid")

# Aggregate at chapter level
chapter_df = df.groupby(['part', 'chapter_id']).agg({
    'text': ' '.join,
    'pos_ratio_adj_adv_to_verb': 'mean',
    'avg_dependency_depth': 'mean',
    'subclauses_per_sentence': 'mean',
    'repeated_words_per_sentence': 'mean',
    'parallel_ratio': 'mean',
    'compound_sentiment': 'mean',
    'is_single_sentence': 'mean'
}).reset_index()

chapter_df['part_num'] = chapter_df['part'].map({p: i for i, p in enumerate(part_order)})
chapter_df = chapter_df.sort_values(['part_num', 'chapter_id']).reset_index(drop=True)
chapter_df = chapter_df.drop('part_num', axis=1)

chapter_metrics_std = (chapter_df[metrics] - chapter_df[metrics].mean()) / chapter_df[metrics].std()
chapter_metrics_std = chapter_metrics_std.fillna(0)  # Handle NaN values
chapter_metrics_std_smooth = chapter_metrics_std.rolling(window=3, center=True, min_periods=1).mean()
part_boundaries = chapter_df.groupby('part').size().cumsum().shift(1).fillna(0).astype(int)

# Generate trend line plots
plt.figure(figsize=(14,6))
for col, color in zip(metrics[:3], ['blue', 'green', 'orange']):
    sns.lineplot(x=chapter_df.index + 1, y=chapter_metrics_std_smooth[col], 
                 label=col.replace('_', ' '), color=color, errorbar='sd')
plt.axvline(x=19.5, color='red', linestyle='--', alpha=0.5)
plt.axvline(x=29.5, color='red', linestyle='--', alpha=0.5)
plt.xlabel("Chapter Index")
plt.ylabel("Standardized Value")
plt.title("Stylistic Feature Trends (Part 1: POS, Depth, Subclauses)")
plt.xticks(ticks=range(1, len(chapter_df) + 1))
plt.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right', frameon=True, borderaxespad=0, 
           edgecolor=(0.5, 0.5, 0.5, 0.5), facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, "stylistic_features_lineplot_1.tif"), 
            dpi=600, format="tif", bbox_inches="tight")
plt.close()

plt.figure(figsize=(14,6))
for col, color in zip(metrics[3:], ['purple', 'cyan', 'brown']):
    sns.lineplot(x=chapter_df.index + 1, y=chapter_metrics_std_smooth[col], 
                 label=col.replace('_', ' '), color=color, errorbar='sd')
plt.axvline(x=19.5, color='red', linestyle='--', alpha=0.5)
plt.axvline(x=29.5, color='red', linestyle='--', alpha=0.5)
plt.xlabel("Chapter Index")
plt.ylabel("Standardized Value")
plt.title("Stylistic Feature Trends (Part 2: Repetition, Parallelism, Sentiment)")
plt.xticks(ticks=range(1, len(chapter_df) + 1))
plt.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right', frameon=True, borderaxespad=0, 
           edgecolor=(0.5, 0.5, 0.5, 0.5), facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, "stylistic_features_lineplot_2.tif"), 
            dpi=600, format="tif", bbox_inches="tight")
plt.close()

# Generate boxplot for POS ratio
plt.figure(figsize=(8,6))
sns.boxplot(x='part', y='pos_ratio_adj_adv_to_verb', data=df, order=part_order, palette=['blue', 'green', 'orange'], showmeans=False)
plt.xlabel("Novel Part")
plt.ylabel("POS Ratio (Adj/Adv to Verb)")
plt.title("Distribution of POS Ratio by Part", fontproperties={'family': 'Arial', 'size': 12})
for idx, row in results_df.iterrows():
    if 'Mann-Whitney' in row['test']:
        parts_pair = row['test'].split('Mann-Whitney ')[1].split(' vs ')
        if len(parts_pair) == 2:
            p1_name, p2_name = parts_pair
            if p1_name in part_order and p2_name in part_order:
                p1_idx = part_order.index(p1_name)
                p2_idx = part_order.index(p2_name)
                if row['pval_corrected'] < 0.05:
                    mid_x = (p1_idx + p2_idx) / 2
                    y_max = df['pos_ratio_adj_adv_to_verb'].max()
                    plt.text(mid_x, y_max * 1.05, '', ha='center', va='bottom', fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "pos_ratio_boxplot.tif"), dpi=600, format="tif", bbox_inches="tight")
plt.close()
pbar.update(1)

OUTPUT_POS_RATIO = os.path.join(OUTPUT_DIR, "pos_ratio_chapter_data.xlsx")
chapter_pos_df = chapter_df[['part', 'chapter_id', 'pos_ratio_adj_adv_to_verb']].copy()
chapter_pos_df.to_excel(OUTPUT_POS_RATIO, index=False)
print(f"Chapter-level POS ratio data saved to: {OUTPUT_POS_RATIO}")

# Perform clustering
chapter_embeddings = model.encode(chapter_df["text"].tolist(), show_progress_bar=False)
chapter_df[['umap_x', 'umap_y']] = umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=2, random_state=42).fit_transform(chapter_embeddings)

# Function to compute clustering metrics
def compute_metrics(X, labels):
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    chi = calinski_harabasz_score(X, labels)
    return sil, dbi, chi

clustering_results = []
reduction_methods = [('PCA', 50), ('PCA', 100), ('UMAP', 10), ('UMAP', 20), ('t-SNE', 2), ('NoRed', None)]
clustering_algos = ['KMeans', 'HDBSCAN']

for method, dim in reduction_methods:
    if method == 'PCA':
        effective_dim = min(dim, chapter_embeddings.shape[0] - 1, chapter_embeddings.shape[1])
        coords = PCA(n_components=effective_dim, random_state=42).fit_transform(chapter_embeddings)
        dim_label = f"{effective_dim}D"
    elif method == 'UMAP':
        coords = umap.UMAP(n_components=dim, n_neighbors=15, random_state=42).fit_transform(chapter_embeddings)
        dim_label = f"{dim}D"
    elif method == 't-SNE':
        coords = TSNE(n_components=dim, perplexity=30, random_state=42).fit_transform(chapter_embeddings)
        dim_label = f"{dim}D"
    else:
        coords = chapter_embeddings
        dim_label = "Original"

    for algo in clustering_algos:
        if algo == 'KMeans':
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(coords)
            labels = kmeans.labels_
        else:
            hdb = hdbscan.HDBSCAN(min_cluster_size=2)
            labels = hdb.fit_predict(coords)
        sil, dbi, chi = compute_metrics(coords, labels)
        clustering_results.append([f"{method} ({dim_label})" if dim else method, algo, sil, dbi, chi])

        if method == 'UMAP' and dim == 10 and algo == 'HDBSCAN':
            chapter_df['hdbscan_umap10_labels'] = labels
            umap_10d_coords = coords

clustering_df = pd.DataFrame(clustering_results, columns=['Dimensionality Reduction', 'Clustering', 'Silhouette', 'DBI', 'CHI'])
clustering_df = clustering_df.round(4)
pbar.update(1)

# Generate UMAP trajectory plot
chapter_df_sorted = chapter_df.sort_values(['part', 'chapter_id'])
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='umap_x', y='umap_y',
    hue='part',
    hue_order=part_order,
    palette=['blue', 'green', 'orange'],
    s=100,
    data=chapter_df_sorted
)
plt.plot(chapter_df_sorted['umap_x'], chapter_df_sorted['umap_y'], color='grey', linewidth=1, alpha=0.5)
plt.xlabel("UMAP X")
plt.ylabel("UMAP Y")
plt.title("Chapter-Level UMAP Stylistic Trajectory")
plt.legend(title="Part", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=False)
plt.savefig(os.path.join(OUTPUT_DIR, "umap_trajectory_chapter.tif"), dpi=600, format="tif", bbox_inches="tight")
plt.close()

# Save results
df.to_excel(os.path.join(OUTPUT_DIR, "stylistic_features_full.xlsx"), index=False, engine='openpyxl')
chapter_df.to_excel(os.path.join(OUTPUT_DIR, "chapter_stylistic_features.xlsx"), index=False, engine='openpyxl')

summary_df = df.groupby('part')[metrics].mean().reset_index()
summary_df = summary_df.round(4)

anova_kruskal_df = results_df[results_df['test'].isin(['ANOVA', 'Kruskal'])][['feature', 'test', 'stat', 'pval', 'pval_corrected']].round(4)
mannwhitney_df = results_df[results_df['test'].str.contains('Mann-Whitney') & (results_df['pval_corrected'] < 0.05)][['feature', 'test', 'stat', 'pval', 'pval_corrected', 'eff_size']].round(4)

excel_path = os.path.join(OUTPUT_DIR, "results_for_paper.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    desc_combined_df.to_excel(writer, sheet_name='Descriptive_Stats_Combined', index=False)
    anova_kruskal_df.to_excel(writer, sheet_name='Statistical_Tests_ANOVA_Kruskal', index=False)
    mannwhitney_df.to_excel(writer, sheet_name='Statistical_Tests_MannWhitney', index=False)
    clustering_df.to_excel(writer, sheet_name='Clustering_Metrics', index=False)
    summary_df.to_excel(writer, sheet_name='Feature_Summary', index=False)
    assumption_tests_df.to_excel(writer, sheet_name='Assumption_Tests', index=False)

print("Paper-ready Excel saved to:", excel_path)
print("Analysis completed! All results saved to:", OUTPUT_DIR)

pbar.close()