import os
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from scipy.stats import f_oneway, kruskal, mannwhitneyu, spearmanr, skew, kurtosis, shapiro, levene, t
from statsmodels.stats.multitest import multipletests
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import re
from nltk.corpus import wordnet as wn
import unicodedata
from collections import Counter
from itertools import combinations

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Limit threads to mitigate memory leak
os.environ["OMP_NUM_THREADS"] = "1"

# Configure font settings for visualizations
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# Apply Seaborn theme
sns.set_theme(style='whitegrid', palette='viridis')

# Define input and output paths
INPUT_FILE = "Narrative_Analysis_output/text_segmentation/to_the_lighthouse_paragraphs.csv"
OUTPUT_BASE_DIR = "Narrative_Analysis_output/narrative_pace"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
OUTPUT_CHAPTER_CSV = os.path.join(OUTPUT_BASE_DIR, "chapter_pace_features.xlsx")
OUTPUT_STATS = os.path.join(OUTPUT_BASE_DIR, "statistical_tests.xlsx")
OUTPUT_DESCRIPTIVE_STATS = os.path.join(OUTPUT_BASE_DIR, "descriptive_stats.xlsx")
OUTPUT_MAIN_TABLE = os.path.join(OUTPUT_BASE_DIR, "main_table.xlsx")
OUTPUT_MANNWHITNEY = os.path.join(OUTPUT_BASE_DIR, "significant_mannwhitney.xlsx")
OUTPUT_BOXPLOT_DATA = os.path.join(OUTPUT_BASE_DIR, "event_perspective_boxplot_data.xlsx")
OUTPUT_SCATTER_DATA = os.path.join(OUTPUT_BASE_DIR, "scatter_points_data.xlsx")

df = pd.read_csv(INPUT_FILE)
chapter_df = df.groupby(['part', 'chapter_id']).agg({'text': lambda x: ' '.join(x)}).reset_index()

part_order = ['The Window', 'Time Passes', 'The Lighthouse']
chapter_df['part'] = pd.Categorical(chapter_df['part'], categories=part_order, ordered=True)
chapter_df = chapter_df.sort_values(by=['part', 'chapter_id']).reset_index(drop=True)

total_steps = 1 + len(chapter_df) + 3 + 2 + 4 + 3 + 3 + 1 + 4 + 1
progress_bar = tqdm(total=total_steps, desc="Narrative Pace Analysis")

# Assign chapter indices
chapter_df['chapter_index'] = range(1, len(chapter_df) + 1)
progress_bar.update(1)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('all-mpnet-base-v2')

# Function to preprocess text
def preprocess_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'[;—]+', '; ', text)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to extract narrative pace features
def calculate_features(text, prev_text=None):
    text = preprocess_text(text)
    doc = nlp(text)
    sentences = sent_tokenize(text)
    num_words = len([t.text for t in doc if t.is_alpha])
    num_sentences = len(sentences)
    avg_sentence_length = num_words / max(num_sentences, 1)

    # Semantic theme score
    theme_references = {
        'time': "The passage of time, years elapsed, seasons changed silently.",
        'memory': "She remembered the past, fleeting thoughts of youth and loss.",
        'psychology': "Her mind wandered, thoughts swirling in endless introspection.",
        'rhythm': "Waves of emotion, breaking and tumbling in the mind."
    }
    text_embedding = bert_model.encode([text])[0]
    theme_embeddings = {k: bert_model.encode([v])[0] for k, v in theme_references.items()}
    theme_scores = {k: cosine_similarity([text_embedding], [v])[0][0] for k, v in theme_embeddings.items()}
    semantic_theme_score = np.mean(list(theme_scores.values()))

    # Event density
    events = []
    psychological_verbs = ['think', 'feel', 'remember', 'imagine', 'reflect', 'ponder']
    for sent in doc.sents:
        verbs = [t for t in sent if t.pos_ == 'VERB' and t.dep_ in ['ROOT', 'xcomp', 'ccomp', 'advcl']]
        entities = [ent.text for ent in sent.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']]
        psych_verb = any(t.lemma_.lower() in psychological_verbs for t in sent)
        if (verbs and entities) or psych_verb:
            events.append(sent.text)
    event_density = len(events) / max(len(sentences), 1)

    # Perspective shift
    perspective_shift = 0
    if prev_text:
        prev_embedding = bert_model.encode([preprocess_text(prev_text)])[0]
        curr_embedding = text_embedding
        perspective_shift += 1 - cosine_similarity([curr_embedding], [prev_embedding])[0][0]
    for sent in sentences:
        sent_doc = nlp(sent)
        pronouns = [t.text.lower() for t in sent_doc if t.pos_ == 'PRON']
        if len(pronouns) > 1:
            perspective_shift += sum(1 for i in range(1, len(pronouns)) if pronouns[i] != pronouns[i-1]) / max(len(sentences), 1)

    # Psychological density
    psych_words = ['mind', 'thought', 'feeling', 'memory', 'dream', 'silence', 'darkness', 'shadow', 'introspection', 'wave']
    psych_synonyms = set()
    for word in psych_words:
        for syn in wn.synsets(word):
            psych_synonyms.update(syn.lemma_names())
    psych_tokens = [t for t in doc if t.lemma_.lower() in psych_synonyms]
    psychological_density = len(psych_tokens) / max(len(doc), 1)

    # Temporal density
    temporal_words = ['time', 'year', 'season', 'past', 'future', 'day', 'night', 'moment', 'hour', 'age']
    temporal_synonyms = set()
    for word in temporal_words:
        for syn in wn.synsets(word):
            temporal_synonyms.update(syn.lemma_names())
    temporal_tokens = [t for t in doc if t.lemma_.lower() in temporal_synonyms]
    temporal_density = len(temporal_tokens) / max(num_words, 1)

    # Narrative movement classification
    sent_class = []
    for sent in doc.sents:
        sent_text = sent.text
        has_dialogue = '"' in sent_text or "'" in sent_text
        verbs = [t for t in sent if t.pos_ == 'VERB']
        if not verbs:
            sent_class.append('pause')
            continue
        past = sum(1 for v in verbs if v.tag_ in ['VBD', 'VBN'])
        present = sum(1 for v in verbs if v.tag_ in ['VBP', 'VBZ', 'VBG'])
        total_v = len(verbs)
        if total_v == 0:
            continue
        past_prop = past / total_v
        if has_dialogue and present > past:
            sent_class.append('scene')
        elif past_prop > 0.6:
            sent_class.append('summary')
        else:
            sent_class.append('ellipsis')
    counter = Counter(sent_class)
    total_sents = len(sent_class) or 1
    props = {}
    for k in ['scene', 'summary', 'pause', 'ellipsis']:
        props[f'prop_{k}'] = counter.get(k, 0) / total_sents

    return avg_sentence_length, semantic_theme_score, event_density, perspective_shift, psychological_density, temporal_density, props

# Extract features at chapter level
chapter_features = []
chapter_movements = []
for idx, row in chapter_df.iterrows():
    prev_text = chapter_df['text'].iloc[idx-1] if idx > 0 and chapter_df['part'].iloc[idx] == chapter_df['part'].iloc[idx-1] else None
    avg_len, sem_theme, event_density, persp_shift, psych_density, temp_density, props = calculate_features(row['text'], prev_text=prev_text)
    chapter_features.append([avg_len, sem_theme, event_density, persp_shift, psych_density, temp_density])
    chapter_movements.append(props)
    progress_bar.update(1)

chapter_features_df = pd.DataFrame(chapter_features, columns=['avg_sentence_length', 'semantic_theme_score', 'event_density', 
                                                             'perspective_shift', 'psychological_density', 'temporal_density'])
chapter_df = pd.concat([chapter_df, chapter_features_df], axis=1)

movements_df = pd.DataFrame(chapter_movements)
chapter_df = pd.concat([chapter_df, movements_df], axis=1)

# Standardize features
scaler_chapter = StandardScaler()
chapter_scaled = scaler_chapter.fit_transform(chapter_df[['avg_sentence_length', 'semantic_theme_score', 'event_density', 
                                                         'perspective_shift', 'psychological_density', 'temporal_density']])
chapter_df[['scaled_avg_sentence_length', 
            'scaled_semantic_theme_score', 
            'scaled_event_density', 
            'scaled_perspective_shift', 
            'scaled_psychological_density', 
            'scaled_temporal_density']] = chapter_scaled
progress_bar.update(1)

# Compute embeddings
chapter_embeddings = bert_model.encode(chapter_df['text'].tolist())
progress_bar.update(1)

# Save chapter features
chapter_df.to_excel(OUTPUT_CHAPTER_CSV, index=False)
progress_bar.update(1)

# Define metrics for analysis
metrics = [
    'avg_sentence_length', 'semantic_theme_score', 'event_density', 
    'perspective_shift', 'psychological_density', 'temporal_density',
    'prop_scene', 'prop_summary', 'prop_pause', 'prop_ellipsis'
]

# Function to compute 95% CI
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

# Compute overall descriptive statistics
overall_stats = []
assumption_tests = []
for metric in metrics:
    data = chapter_df[metric].dropna()
    if len(data) == 0:
        continue
    mean = data.mean()
    sd = data.std()
    median = data.median()
    iqr = data.quantile(0.75) - data.quantile(0.25)
    skewness = skew(data, nan_policy='omit')
    kurt = kurtosis(data, nan_policy='omit')
    ci_lower, ci_upper = calculate_ci(data)
    overall_stats.append({
        'Metric': metric.replace('_', ' '),
        'Mean': mean,
        'Std': sd,
        'Median': median,
        'IQR': iqr,
        'Skewness': skewness,
        'Kurtosis': kurt,
        '95% CI Lower': ci_lower,
        '95% CI Upper': ci_upper
    })
    
    # Shapiro-Wilk test for normality
    if len(data) >= 3:
        shapiro_stat, shapiro_p = shapiro(data)
        assumption_tests.append({
            'Metric': metric.replace('_', ' '),
            'Test': 'Shapiro-Wilk (Overall)',
            'Statistic': shapiro_stat,
            'p-value': shapiro_p
        })

overall_stats_df = pd.DataFrame(overall_stats).round(4)
progress_bar.update(1)

# Compute grouped descriptive statistics
key_metrics = ['avg_sentence_length', 'event_density', 'psychological_density', 'prop_pause']
grouped_stats = []
for part, group in chapter_df.groupby('part'):
    for metric in key_metrics:
        data = group[metric].dropna()
        if len(data) == 0:
            continue
        mean = data.mean()
        sd = data.std()
        median = data.median()
        ci_lower, ci_upper = calculate_ci(data)
        grouped_stats.append({
            'Part': part,
            'Metric': metric.replace('_', ' '),
            'Mean': mean,
            'Std': sd,
            'Median': median,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper
        })
grouped_stats_df = pd.DataFrame(grouped_stats).round(4)
progress_bar.update(1)

# Grouped assumption tests
for metric in metrics:
    groups = [chapter_df[chapter_df['part'] == p][metric].dropna() for p in part_order if len(chapter_df[chapter_df['part'] == p]) > 0]
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        levene_stat, levene_p = levene(*groups)
        assumption_tests.append({
            'Metric': metric.replace('_', ' '),
            'Test': 'Levene (Homogeneity)',
            'Statistic': levene_stat,
            'p-value': levene_p
        })
    for i, group_data in enumerate(groups):
        if len(group_data) >= 3:
            shapiro_stat, shapiro_p = shapiro(group_data)
            assumption_tests.append({
                'Metric': metric.replace('_', ' '),
                'Test': f'Shapiro-Wilk ({part_order[i]})',
                'Statistic': shapiro_stat,
                'p-value': shapiro_p
            })

assumption_tests_df = pd.DataFrame(assumption_tests).round(4)

# Compute correlation matrix
corr_matrix = chapter_df[metrics[:6]].corr(method='spearman').round(2)
corr_matrix.index = [col.replace('_', ' ') for col in corr_matrix.index]
corr_matrix.columns = [col.replace('_', ' ') for col in corr_matrix.columns]
corr_matrix_df = corr_matrix.reset_index().rename(columns={'index': 'Metric'})
progress_bar.update(1)

# Save descriptive statistics
with pd.ExcelWriter(OUTPUT_DESCRIPTIVE_STATS) as writer:
    overall_stats_df.to_excel(writer, sheet_name='Overall_Statistics', index=False)
    grouped_stats_df.to_excel(writer, sheet_name='Grouped_Statistics', index=False)
    corr_matrix_df.to_excel(writer, sheet_name='Correlation_Matrix', index=False)
    assumption_tests_df.to_excel(writer, sheet_name='Assumption_Tests', index=False)

parts = chapter_df['part'].unique()
stats_results = {}

# Function to compute Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    diff = x.mean() - y.mean()
    pooled_std = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return diff / pooled_std if pooled_std != 0 else 0

# Perform ANOVA
stats_results['ANOVA'] = {
    'Average Sentence Length': f_oneway(*[chapter_df[chapter_df['part'] == p]['avg_sentence_length'] for p in parts]),
    'Semantic Theme Score': f_oneway(*[chapter_df[chapter_df['part'] == p]['semantic_theme_score'] for p in parts]),
    'Event Density': f_oneway(*[chapter_df[chapter_df['part'] == p]['event_density'] for p in parts]),
    'Perspective Shift': f_oneway(*[chapter_df[chapter_df['part'] == p]['perspective_shift'] for p in parts]),
    'Psychological Density': f_oneway(*[chapter_df[chapter_df['part'] == p]['psychological_density'] for p in parts]),
    'Temporal Density': f_oneway(*[chapter_df[chapter_df['part'] == p]['temporal_density'] for p in parts])
}
progress_bar.update(1)

# Perform Kruskal-Wallis
stats_results['Kruskal-Wallis'] = {
    'Average Sentence Length': kruskal(*[chapter_df[chapter_df['part'] == p]['avg_sentence_length'] for p in parts]),
    'Semantic Theme Score': kruskal(*[chapter_df[chapter_df['part'] == p]['semantic_theme_score'] for p in parts]),
    'Event Density': kruskal(*[chapter_df[chapter_df['part'] == p]['event_density'] for p in parts]),
    'Perspective Shift': kruskal(*[chapter_df[chapter_df['part'] == p]['perspective_shift'] for p in parts]),
    'Psychological Density': kruskal(*[chapter_df[chapter_df['part'] == p]['psychological_density'] for p in parts]),
    'Temporal Density': kruskal(*[chapter_df[chapter_df['part'] == p]['temporal_density'] for p in parts])
}
progress_bar.update(1)

# Perform Mann-Whitney U tests
mann_results = []
for g1, g2 in combinations(parts, 2):
    u_event, p_event = mannwhitneyu(chapter_df[chapter_df['part'] == g1]['event_density'], 
                                    chapter_df[chapter_df['part'] == g2]['event_density'])
    eff_event = cohens_d(chapter_df[chapter_df['part'] == g1]['event_density'], 
                         chapter_df[chapter_df['part'] == g2]['event_density'])
    u_persp, p_persp = mannwhitneyu(chapter_df[chapter_df['part'] == g1]['perspective_shift'], 
                                    chapter_df[chapter_df['part'] == g2]['perspective_shift'])
    eff_persp = cohens_d(chapter_df[chapter_df['part'] == g1]['perspective_shift'], 
                         chapter_df[chapter_df['part'] == g2]['perspective_shift'])
    u_psych, p_psych = mannwhitneyu(chapter_df[chapter_df['part'] == g1]['psychological_density'], 
                                    chapter_df[chapter_df['part'] == g2]['psychological_density'])
    eff_psych = cohens_d(chapter_df[chapter_df['part'] == g1]['psychological_density'], 
                         chapter_df[chapter_df['part'] == g2]['psychological_density'])
    u_temp, p_temp = mannwhitneyu(chapter_df[chapter_df['part'] == g1]['temporal_density'], 
                                  chapter_df[chapter_df['part'] == g2]['temporal_density'])
    eff_temp = cohens_d(chapter_df[chapter_df['part'] == g1]['temporal_density'], 
                        chapter_df[chapter_df['part'] == g2]['temporal_density'])
    mann_results.append({
        'Comparison': f"{g1} vs {g2}",
        'Feature': 'Event Density',
        'Statistic': u_event,
        'p-value': p_event,
        'Effect Size': eff_event
    })
    mann_results.append({
        'Comparison': f"{g1} vs {g2}",
        'Feature': 'Perspective Shift',
        'Statistic': u_persp,
        'p-value': p_persp,
        'Effect Size': eff_persp
    })
    mann_results.append({
        'Comparison': f"{g1} vs {g2}",
        'Feature': 'Psychological Density',
        'Statistic': u_psych,
        'p-value': p_psych,
        'Effect Size': eff_psych
    })
    mann_results.append({
        'Comparison': f"{g1} vs {g2}",
        'Feature': 'Temporal Density',
        'Statistic': u_temp,
        'p-value': p_temp,
        'Effect Size': eff_temp
    })
mann_df = pd.DataFrame(mann_results)
mann_df['Corrected p-value'] = multipletests(mann_df['p-value'], method='fdr_bh')[1]
mann_df = mann_df[mann_df['Corrected p-value'] < 0.05].round(4)
progress_bar.update(1)

part_order = ['The Window', 'Time Passes', 'The Lighthouse']

# Generate boxplots for event density and perspective shift
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='part', y='scaled_event_density', data=chapter_df, palette='viridis', order=part_order, showmeans=False)
plt.title("Event Density by Part")
plt.xlabel("Narrative Part")
plt.ylabel("Standardized Event Density")
plt.subplot(1, 2, 2)
sns.boxplot(x='part', y='scaled_perspective_shift', data=chapter_df, palette='viridis', order=part_order, showmeans=False)
plt.title("Perspective Shift by Part")
plt.xlabel("Narrative Part")
plt.ylabel("Standardized Perspective Shift")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "event_perspective_boxplot.tif"), format='tiff', dpi=600)
plt.close()

boxplot_data = chapter_df[['part', 'chapter_id', 'scaled_event_density', 'scaled_perspective_shift']].copy()
boxplot_data.to_excel(OUTPUT_BOXPLOT_DATA, index=False)
print(f"Boxplot data saved to: {OUTPUT_BOXPLOT_DATA}")
progress_bar.update(1)

# Generate scatter plot for event and psychological density
part_markers = {'The Window': 'o', 'Time Passes': '^', 'The Lighthouse': 'x'}
part_colors = {'The Window': '#008000', 'Time Passes': '#FF4500', 'The Lighthouse': '#0000CD'}
plt.figure(figsize=(14, 10))
for part, marker in part_markers.items():
    subset = chapter_df[chapter_df['part'] == part]
    x_jitter = subset['scaled_event_density'] + np.random.normal(0, 0.05, len(subset))
    y_jitter = subset['scaled_psychological_density'] + np.random.normal(0, 0.05, len(subset))
    plt.scatter(
        x_jitter,
        y_jitter,
        label=part,
        marker=marker,
        c=part_colors[part],
        s=70,
        alpha=0.7,
        edgecolors='white'
    )

sns.kdeplot(
    data=chapter_df, 
    x='scaled_event_density', 
    y='scaled_psychological_density', 
    levels=5, 
    color='gray', 
    thresh=0.2, 
    linewidths=1
)

sns.regplot(
    x='scaled_event_density',
    y='scaled_psychological_density',
    data=chapter_df,
    scatter=False,
    color='black',
    line_kws={'linewidth': 1, 'linestyle': '--'}
)

plt.title("Relationship Between Event Density and Psychological Density in Narrative Parts", fontsize=18)
plt.xlabel("Standardized Event Density", fontsize=16)
plt.ylabel("Standardized Psychological Density", fontsize=16)
plt.legend(title="Narrative Part", fontsize=14, title_fontsize=16)
plt.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "event_psychological_scatter_reg.tif"), format='tiff', dpi=600)
plt.close()
progress_bar.update(1)

# Generate scatter plot with chapter labels
plt.figure(figsize=(14, 10))
for part, marker in part_markers.items():
    subset = chapter_df[chapter_df['part'] == part].sort_values('chapter_index')
    
    x_jitter = subset['scaled_event_density'] + np.random.normal(0, 0.05, len(subset))
    y_jitter = subset['scaled_psychological_density'] + np.random.normal(0, 0.05, len(subset))
    
    plt.scatter(
        x_jitter,
        y_jitter,
        label=part,
        marker=marker,
        c=part_colors[part],
        s=70,
        alpha=0.7,
        edgecolors='white'
    )

    for xi, yi, ch_idx in zip(x_jitter, y_jitter, subset['chapter_index']):
        plt.text(xi + 0.01, yi + 0.01, str(ch_idx), fontsize=9, alpha=0.8)

sns.kdeplot(
    data=chapter_df, 
    x='scaled_event_density', 
    y='scaled_psychological_density', 
    levels=5, 
    color='gray', 
    thresh=0.2, 
    linewidths=1
)

sns.regplot(
    x='scaled_event_density',
    y='scaled_psychological_density',
    data=chapter_df,
    scatter=False,
    color='black',
    line_kws={'linewidth': 1, 'linestyle': '--'}
)

plt.title("Event vs Psychological Density with Chapter Labels", fontsize=18)
plt.xlabel("Standardized Event Density", fontsize=16)
plt.ylabel("Standardized Psychological Density", fontsize=16)
plt.legend(title="Narrative Part", fontsize=14, title_fontsize=16)
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "event_psychological_scatter_reg_with_labels.tif"), format='tiff', dpi=600)
plt.close()

scatter_data = chapter_df[['part', 'chapter_index', 'scaled_event_density', 'scaled_psychological_density']].copy()
scatter_data['coordinate'] = '(' + scatter_data['scaled_event_density'].round(4).astype(str) + ', ' + scatter_data['scaled_psychological_density'].round(4).astype(str) + ')'
scatter_data = scatter_data[['part', 'chapter_index', 'coordinate']]
scatter_data.to_excel(OUTPUT_SCATTER_DATA, index=False)
print(f"Scatter points data saved to: {OUTPUT_SCATTER_DATA}")
progress_bar.update(1)

# Generate correlation heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='viridis', vmin=-1, vmax=1, center=0)
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        value = corr_matrix.iloc[i, j]
        if np.isnan(value):
            continue
        text_color = 'white' if value < 0 else 'black'
        ax.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=12)
plt.title("Spearman Correlation Matrix of Narrative Pace Features", fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "correlation_heatmap.tif"), format='tiff', dpi=600)
plt.close()
progress_bar.update(1)

# Generate line plots for trends
plt.figure(figsize=(16, 6))
chapter_df['smoothed_event_density'] = chapter_df['scaled_event_density'].rolling(window=3, min_periods=1).mean()
chapter_df['smoothed_perspective_shift'] = chapter_df['scaled_perspective_shift'].rolling(window=3, min_periods=1).mean()
sns.lineplot(x='chapter_index', y='scaled_avg_sentence_length', data=chapter_df, label='Average Sentence Length', color='#32CD32', errorbar='sd')
sns.lineplot(x='chapter_index', y='smoothed_event_density', data=chapter_df, label='Event Density (Smoothed)', color='#1E90FF', errorbar='sd')
sns.lineplot(x='chapter_index', y='smoothed_perspective_shift', data=chapter_df, label='Perspective Shift (Smoothed)', color='#FF8C00', errorbar='sd')
for idx in chapter_df[chapter_df['part'] != chapter_df['part'].shift()].index:
    plt.axvline(x=chapter_df.loc[idx, 'chapter_index'] - 0.5, color='#FF0000', linestyle='--', alpha=0.8)
plt.title("Temporal Trends in Standardized Narrative Pace Features Across Chapters", fontsize=16)
plt.xlabel("Chapter Index")
plt.ylabel("Standardized Feature Values")
plt.xticks(np.arange(1, 44, 1), fontsize=12, rotation=0)
plt.legend(title="Feature", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "pace_trends_chapter_2.tif"), format='tiff', dpi=600)
plt.close()
progress_bar.update(1)

plt.figure(figsize=(16, 6))
sns.lineplot(x='chapter_index', y='scaled_semantic_theme_score', data=chapter_df, label='Semantic Theme Score', color='#800080', errorbar='sd')
sns.lineplot(x='chapter_index', y='scaled_psychological_density', data=chapter_df, label='Psychological Density', color='#FF4500', errorbar='sd')
sns.lineplot(x='chapter_index', y='scaled_temporal_density', data=chapter_df, label='Temporal Density', color='#20B2AA', errorbar='sd')
for idx in chapter_df[chapter_df['part'] != chapter_df['part'].shift()].index:
    plt.axvline(x=chapter_df.loc[idx, 'chapter_index'] - 0.5, color='#FF0000', linestyle='--', alpha=0.8)
plt.title("Temporal Trends in Standardized Narrative Pace Features Across Chapters", fontsize=16)
plt.xlabel("Chapter Index")
plt.ylabel("Standardized Feature Values")
plt.xticks(np.arange(1, 44, 1), fontsize=12, rotation=0)
plt.legend(title="Feature", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "pace_trends_chapter_1.tif"), format='tiff', dpi=600)
plt.close()
progress_bar.update(1)

# Generate stacked bar chart for narrative movements
movement_cols = ['prop_scene', 'prop_summary', 'prop_pause', 'prop_ellipsis']
chapter_df[movement_cols] = chapter_df[movement_cols].fillna(0)
plt.figure(figsize=(16, 8))
chapter_df.plot(x='chapter_index', kind='bar', stacked=True, y=movement_cols, colormap='viridis', figsize=(16, 8))
plt.title("Proportions of Narrative Movement Types Across Chapters", fontsize=16)
plt.xlabel("Chapter Index")
plt.ylabel("Proportion")
plt.xticks(np.arange(0, len(chapter_df), 1), np.arange(1, len(chapter_df) + 1), fontsize=12, rotation=0)
plt.legend(title="Movement Type", loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE_DIR, "narrative_movements_chapter.tif"), format='tiff', dpi=600)
plt.close()

OUTPUT_MOVEMENT_DATA = os.path.join(OUTPUT_BASE_DIR, "narrative_movements_data.xlsx")
movement_data = chapter_df[['part', 'chapter_id', 'chapter_index', 
                            'prop_scene', 'prop_summary', 'prop_pause', 'prop_ellipsis']].copy()
movement_data.to_excel(OUTPUT_MOVEMENT_DATA, index=False)
progress_bar.update(1)

# Save statistical results
with pd.ExcelWriter(OUTPUT_STATS) as writer:
    anova_df = pd.DataFrame({
        k: [v.statistic, v.pvalue] for k, v in stats_results['ANOVA'].items()
    }, index=['Statistic', 'p-value']).round(4)
    anova_df.columns = [col.replace('_', ' ') for col in anova_df.columns]
    anova_df.to_excel(writer, sheet_name='ANOVA')
    kruskal_df = pd.DataFrame({
        k: [v.statistic, v.pvalue] for k, v in stats_results['Kruskal-Wallis'].items()
    }, index=['Statistic', 'p-value']).round(4)
    kruskal_df.columns = [col.replace('_', ' ') for col in kruskal_df.columns]
    kruskal_df.to_excel(writer, sheet_name='Kruskal-Wallis')
    mann_df.to_excel(writer, sheet_name='Mann-Whitney U', index=False)

main_table = chapter_df.groupby('part')[metrics[:6] + ['prop_pause']].mean().reset_index().round(4)
main_table.columns = [col.replace('_', ' ') for col in main_table.columns]
main_table.to_excel(OUTPUT_MAIN_TABLE, index=False)
mann_df.to_excel(OUTPUT_MANNWHITNEY, index=False)
progress_bar.update(1)

progress_bar.close()

print("\nNarrative Pace Analysis Completed Successfully!")
print(f"Results saved to: {OUTPUT_BASE_DIR}\n")