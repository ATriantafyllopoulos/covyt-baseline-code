import argparse
import audmetric
import audformat
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn import metrics

def auc(true, pred):
    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label='negative')
    # print(fpr)
    # print(tpr)
    # exit()
    return float(metrics.auc(fpr, tpr))

def get_CI(preds, labels, result, metric):

    results = []
    for s in range(1000):
        np.random.seed(s)
        sample = np.random.choice(range(len(preds)), len(preds), replace=True) #boost with replacement
        sample_preds = preds[sample]
        sample_labels = labels[sample]

        res = metric(sample_labels, sample_preds)
        results.append(res)

    q_0 = pd.DataFrame(np.array(results)).quantile(0.025)[0] #2.5% percentile
    q_1 = pd.DataFrame(np.array(results)).quantile(0.975)[0] #97.5% percentile

    return(f'{result*100:.1f} ({q_0*100:.1f}--{q_1*100:.1f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate")
    parser.add_argument('root')
    parser.add_argument(
        '--dataset',
        help='Path to dataset in \'audformat\'',
        default='/nas/staff/data_work/Andreas/COVID-19_YouTube/dataset'
    )
    parser.add_argument(
        '--partitioning',
        default=[
            'language.disjoint',
            'speaker.disjoint',
            'file.disjoint',
            'speaker.inclusive'
        ],
        choices=[
            'language.disjoint',
            'speaker.disjoint',
            'file.disjoint',
            'speaker.inclusive'
        ],
        nargs='+'
    )
    parser.add_argument(
        '--normalization',
        default=[
            'standard',
        ],
        choices=[
            'standard',
            'speaker',
            'speaker-negative',
        ],
        nargs='+'
    )
    parser.add_argument(
        '--metric',
        default='AUC',
        choices=[
            'AUC',
            'UAR'
        ]
    )
    args = parser.parse_args()
    
    db = audformat.Database.load(args.dataset)

    if args.metric == 'AUC':
        metric = auc
        column = 'probabilities'
    elif args.metric == 'UAR':
        metric = audmetric.unweighted_average_recall
        column = 'predictions'
    else:
        raise NotImplementedError(args.metric)

    mean_fpr = np.linspace(0, 1, 100)
    sns.set_context('paper')
    plt.rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans'], 'size': 14})
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    feature_dicts = {
        'ComParE': {
            'label': 'ComParE',
            'color': 'indigo',
            'style': '--'
        },
        'eGeMAPSv02': {
            'label': 'eGeMAPS',
            'color': 'steelblue',
            'style': '-.'
        },
        'wav2vec2-large-xlsr-53-features': {
            'label': 'w2v2-xlsr',
            'color': 'firebrick',
            'style': '-'
            
        },
    } 

    for partitioning in args.partitioning:
        print(partitioning)
        print('=' * 10)
        features = os.listdir(os.path.join(args.root, partitioning))

        fig, ax = plt.subplots(figsize=[6, 4])
        
        for feature in features:
            print(feature)
            print('-' * 10)
            df = pd.read_csv(os.path.join(args.root, f'{partitioning}', feature, 'results.csv'))
            df['start'] = df['start'].apply(pd.to_timedelta)
            df['end'] = df['end'].apply(pd.to_timedelta)
            df.set_index(['file', 'start', 'end'], inplace=True)
            folds = sorted(list(set([x.split('.')[-2] for x in db.tables if f'folds.{partitioning}' in x])))
            results = []
            file_results = []
            tprs = []
            for fold in folds:
                test = db[f'folds.{partitioning}.{fold}.test'].df
                results.append(
                    metric(
                        df.reindex(test.index)['covid'],
                        df.reindex(test.index)[column]
                    )
                )
                fold_df = df.reindex(test.index).reset_index()
                file_df = fold_df.groupby('file').apply(
                    lambda x: pd.Series({
                        'covid': x['covid'].mode()[0],
                        column: x[column].mode()[0] if args.metric == 'UAR' else x[column].mean()
                    })
                )
                file_results.append(
                    metric(
                        file_df['covid'],
                        file_df[column]
                    )
                )
                ci = get_CI(df.reindex(test.index)[column], df.reindex(test.index)["covid"], results[-1], metric)
                file_ci = get_CI(file_df[column], file_df["covid"], file_results[-1], metric)
                print(f'Fold {fold} & {ci} & {file_ci}')

                fpr, tpr, thresholds = metrics.roc_curve(file_df['covid'], file_df[column], pos_label='negative')
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.plot(
                mean_fpr,
                mean_tpr,
                label=feature_dicts[feature]['label'],
                color=feature_dicts[feature]['color'],
                linestyle=feature_dicts[feature]['style'],
                lw=2
            )
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color=feature_dicts[feature]['color'],
                alpha=0.2,
                linestyle=feature_dicts[feature]['style'],
            )
            print((f'{partitioning} & {feature} & '
                   f'{np.mean(results) * 100:.1f} ({np.std(results) * 100:.1f}) & '
                   f'{np.mean(file_results) * 100:.1f} ({np.std(file_results) * 100:.1f})'
            ))
        print()
        sns.despine(ax=ax)
        plt.legend(loc='lower right')
        plt.title('Receiver operating characteristic (speaker disjoint)', fontdict={'size': 12})
        plt.xlabel('False Positive Rate', fontdict={'size': 12})
        plt.ylabel('True Positive Rate', fontdict={'size': 12})
        plt.grid(axis='y', alpha=0.1, c='k')
        plt.tight_layout()
        plt.savefig(f'{partitioning}.auc.png')
        plt.savefig(f'{partitioning}.auc.pdf')
        plt.close()

# eGeMAPS
# Fold 1 & 51% (56% - 58%) & 60% (51% - 69%)
# Fold 2 & 61% (56% - 58%) & 58% (46% - 69%)
# Fold 3 & 51% (56% - 58%) & 60% (51% - 69%)
# Fold 4 & 60% (56% - 58%) & 56% (50% - 63%)
# Fold 5 & 61% (56% - 58%) & 58% (46% - 69%)
# Fold 6 & 60% (56% - 58%) & 56% (50% - 63%)
# speaker.independent & standard & 57 (4) & 58 (2)

# ComParE
# Fold 1 & 63% (70% - 71%) & 72% (62% - 81%)
# Fold 2 & 73% (70% - 71%) & 64% (53% - 74%)
# Fold 3 & 63% (70% - 71%) & 72% (62% - 81%)
# Fold 4 & 75% (70% - 71%) & 71% (59% - 82%)
# Fold 5 & 73% (70% - 71%) & 64% (53% - 74%)
# Fold 6 & 75% (70% - 71%) & 71% (59% - 82%)
# speaker.independent & standard & 70 (5) & 69 (3)

