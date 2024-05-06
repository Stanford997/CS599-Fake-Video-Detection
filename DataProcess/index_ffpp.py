from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from isplutils.utils import extract_meta_av, extract_meta_cv


def main():
    source_dir = Path('/Users/caozhen/Deepfake-Detection/Datasets')
    videodataset_path = Path('data/ffpp_videos.pkl')
    videodataset_path.parent.mkdir(parents=True, exist_ok=True)

    # DataFrame
    if videodataset_path.exists():
        print('Loading video DataFrame')
        df_videos = pd.read_pickle(videodataset_path)
    else:
        print('Creating video DataFrame')

        ff_videos = Path(source_dir).rglob('*.mp4')
        df_videos = pd.DataFrame(
            {'path': [f.relative_to(source_dir) for f in ff_videos if 'mask' not in str(f) and 'raw' not in str(f)]})

        df_videos['height'] = df_videos['width'] = df_videos['frames'] = np.zeros(len(df_videos), dtype=np.uint16)

        # Extract metadata
        with Pool() as p:
            meta = p.map(extract_meta_av, df_videos['path'].map(lambda x: str(source_dir.joinpath(x))))
        meta = np.stack(meta)
        df_videos.loc[:, ['height', 'width', 'frames']] = meta

        # Fix for videos that av cannot decode properly
        for idx, record in df_videos[df_videos['frames'] == 0].iterrows():
            meta = extract_meta_cv(str(source_dir.joinpath(record['path'])))
            df_videos.loc[idx, ['height', 'width', 'frames']] = meta

        df_videos['class'] = df_videos['path'].map(lambda x: x.parts[0]).astype('category')
        df_videos['label'] = df_videos['class'].map(
            lambda x: True if x == 'manipulated_sequences' else False)  # True is FAKE, False is REAL
        df_videos['source'] = df_videos['path'].map(lambda x: x.parts[1]).astype('category')
        df_videos['quality'] = df_videos['path'].map(lambda x: x.parts[2]).astype('category')
        df_videos['name'] = df_videos['path'].map(lambda x: x.with_suffix('').parts[-1])

        df_videos['original'] = -1 * np.ones(len(df_videos), dtype=np.int16)

        df_videos.loc[(df_videos['label'] == True) & (df_videos['source'] != 'DeepFakeDetection'), 'original'] = \
            df_videos[(df_videos['label'] == True) & (df_videos['source'] != 'DeepFakeDetection')]['name'].map(
                lambda x: df_videos.index[
                    np.flatnonzero(df_videos['name'] == x.split('_')[0])[0]]
            )

        df_videos.loc[(df_videos['label'] == True) & (df_videos['source'] == 'DeepFakeDetection'), 'original'] = \
            df_videos[(df_videos['label'] == True) & (df_videos['source'] == 'DeepFakeDetection')]['name'].map(
                lambda x: df_videos.index[
                    np.flatnonzero(df_videos['name'] == x.split('_')[0] + '__' + x.split('__')[1])[0]]
            )
        # Save as pickle file
        print('Saving video DataFrame to {}'.format(videodataset_path))
        df_videos.to_pickle(str(videodataset_path))

    print('Real videos: {:d}'.format(sum(df_videos['label'] == False)))
    print('Fake videos: {:d}'.format(sum(df_videos['label'] == True)))


if __name__ == '__main__':
    main()
