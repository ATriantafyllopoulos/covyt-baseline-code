import argparse
import audeer
import audformat
import audmetric
import joblib
import os
import pandas as pd
import yaml

from sklearn.svm import (
    SVC
)
from sklearn.pipeline import (
    Pipeline
)
from sklearn.preprocessing import (
    StandardScaler
)


def speaker_normalization(features: pd.DataFrame, df: pd.DataFrame):
    r"""Speaker-specific normalization.

    Args:
        features: dataframe containing all speakers
        df: dataframe with database information
    """
    data = features.values.copy()
    for speaker in df['speaker'].unique():
        indices = df['speaker'] == speaker
        data[indices, :] = StandardScaler().fit_transform(data[indices, :])
    return pd.DataFrame(
        data=data,
        index=features.index,
        columns=features.columns
    )

def speaker_negative_normalization(features: pd.DataFrame, df: pd.DataFrame):
    r"""Speaker-specific normalization using only \'before\' samples.

    \'Before\' samples correspond to samples obtained before
    the speaker was diagnosed with Covid-19.

    Args:
        features: dataframe containing all speakers
        df: dataframe with database information
    """
    data = features.values.copy()
    for speaker in df['speaker'].unique():
        indices = df['speaker'] == speaker
        negative_indices = (df['speaker'] == speaker) & (df['covid'] == False)
        print(negative_indices)
        scaler = StandardScaler()
        scaler.fit(data[negative_indices, :])
        data[indices, :] = scaler.transform(data[indices, :])
    return pd.DataFrame(
        data=data,
        index=features.index,
        columns=features.columns
    )


def cv_training(
    db: audformat.Database,
    partitioning: str,
    features: pd.DataFrame,
    normalization: str,
    root: str
):
    r"""CV training.

    Function performs cross-validation
    for COVID-19 prediction.

    Args:
        df: dataframe with dataset information
        features: dataframe with features
        normalization: normalization process
        root: path to store results
    """

    df = db['covid'].df
    df = df.loc[~df['covid'].isna()]
    df['covid'] = df['covid'].apply(lambda x: 'positive' if x else 'negative')
    df['speaker'] = db['files'].get(index=df.index)['speaker']
    folds = sorted(list(set([x.split('.')[-2] for x in db.tables if f'folds.{partitioning}' in x])))

    metrics = {
        'F1': audmetric.unweighted_average_fscore,
        'UAR': audmetric.unweighted_average_recall,
        'ACC': audmetric.accuracy
    }

    if not os.path.exists(os.path.join(root, 'results.csv')):
        for fold in folds:

            def get_fold(db, fold_name):
                df = db[f'folds.{partitioning}.{fold}.{fold_name}'].df
                df['speaker'] = db['files'].get(index=df.index)['speaker']
                df = df.loc[~df['covid'].isna()]
                df['covid'] = df['covid'].apply(lambda x: 'positive' if x else 'negative')
                return df
            df_train = get_fold(db, 'train')
            df_dev = get_fold(db, 'dev')
            df_test = get_fold(db, 'test')

            features = features.fillna(0)

            c_params = [
                .0001, 
                .0005, 
                .001, 
                .005, 
                .01, 
                .05, 
                .1, 
                .5, 
                1
            ]

            steps = []
            if normalization == 'standard':
                # normalization performed on the fly for each fold
                steps.append(('scale', StandardScaler()))
            steps.append(('classify', SVC(kernel='rbf', probability=True)))

            max_f1 = 0
            best_c = None
            for c_param in audeer.progress_bar(
                c_params,
                total=len(c_params),
                desc='LOSO',
                disable=True
            ):
                
                clf = Pipeline(steps)
                clf.set_params(**{'classify__C': c_param})
                clf.fit(
                    features.loc[df_train.index],
                    df_train['covid'],
                )
                pred = clf.predict(features.loc[df_dev.index])
                f1_score = audmetric.unweighted_average_fscore(df_dev['covid'], pred)
                if f1_score > max_f1:
                    max_f1 = f1_score
                    best_c = c_param
            
            clf.set_params(**{'classify__C': best_c})
            clf.fit(
                features.loc[pd.concat((df_train, df_dev)).index],
                pd.concat((df_train, df_dev))['covid'],
            )
            joblib.dump(
                clf,
                os.path.join(root, f'clf.{fold}.pkl')
            )
            df.loc[df_test.index, 'predictions'] = clf.predict(features.loc[df_test.index])
            df.loc[df_test.index, 'probabilities'] = clf.predict_proba(features.loc[df_test.index])[:, 0]
            
        df.reset_index(inplace=True)
        df.to_csv(os.path.join(root, 'results.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join(root, 'results.csv'))

    results = {
        key: metrics[key](df['covid'], df['predictions'])
        for key in metrics
    }
    with open(os.path.join(root, 'results.yaml'), 'w') as fp:
        yaml.dump(results, fp)

    file_df = df.groupby('file').apply(
        lambda x: pd.Series({
            'covid': x['covid'].mode()[0],
            'predictions': x['predictions'].mode()[0]
        })
    )

    results = {
        key: metrics[key](file_df['covid'], file_df['predictions'])
        for key in metrics
    }
    with open(os.path.join(root, 'speaker_results.yaml'), 'w') as fp:
        yaml.dump(results, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOSO-CV Training for COVID-19')
    parser.add_argument(
        '--dataset',
        help='Path to dataset in \'audformat\''
    )
    parser.add_argument(
        '--features',
        help='Path to features in \'audformat\''
    )
    parser.add_argument(
        '--results',
        default='./results',
        help='Path to store results'
    )
    parser.add_argument(
        '--partitioning',
        default='language.disjoint',
        choices=[
            'language.disjoint',
            'speaker.disjoint',
            'file.disjoint',
            'speaker.inclusive'
        ]
    )
    parser.add_argument(
        '--normalization',
        default='standard',
        choices=[
            'standard',
            'speaker',
            'speaker-negative',
        ]
    )
    args = parser.parse_args()

    ### Load and prepare dataset and features
    db = audformat.Database.load(args.dataset)
    features = pd.read_csv(args.features)
    features['start'] = features['start'].apply(pd.to_timedelta)
    features['end'] = features['end'].apply(pd.to_timedelta)
    features.set_index(['file', 'start', 'end'], inplace=True)

    ### Perform normalization
    if args.normalization == 'standard':
        pass  # done on the fly in each fold
    elif args.normalization == 'speaker':
        features = speaker_normalization(features, db['covid'].df)
    elif args.normalization == 'speaker-negative':
        features = speaker_negative_normalization(features, db['covid'].df)
    
    root = audeer.mkdir(args.results)

    print((
        f'Running training for COVID-19 prediction with '
        f'{args.normalization} normalization '
        f'using {args.partitioning} partitioning.'
    ))
    cv_training(
        db=db,
        partitioning=args.partitioning,
        features=features,
        normalization=args.normalization,
        root=root
    )


    