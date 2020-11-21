import torch
import sys
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
import utils
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import uuid
import argparse
import sed_eval
from models import crnn
import os

SAMPLE_RATE = 22050
EPS = np.spacing(1)
LMS_ARGS = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.02),
    'win_length': int(SAMPLE_RATE * 0.04)
}
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


def extract_feature(wavefilepath, **kwargs):
    try:
        wav, sr = sf.read(wavefilepath, dtype='float32')
    except:
        print("error with file {}".format(wavefilepath))
        wav = np.random.randn(480000)
        sr = 48000

    if len(wav)==0:
        print("error with file {}".format(wavefilepath))
        wav = np.random.randn(480000)
        sr = 48000
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    return np.log(
        librosa.feature.melspectrogram(wav.astype(np.float32), sr, **kwargs) +
        EPS).T


class OnlineLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.dlist = data_list
        self.kwargs = kwargs

    def __getitem__(self, idx):
        return extract_feature(wavefilepath=self.dlist[idx],
                               **self.kwargs), self.dlist[idx]

    def __len__(self):
        return len(self.dlist)


MODELS = {
    'gpvf': {
        'model': crnn,
        'outputdim': 527,
        'encoder': 'label_encoders/gpv_f.pth',
        'pretrained': 'pretrained/gpv_f.pth',
        'resolution': 0.02
    },
    'gpvb': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'label_encoders/gpv_b.pth',
        'pretrained': 'pretrained/gpv_b.pth',
        'resolution': 0.02
    },
    'vadc': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'label_encoders/vad_c.pth',
        'pretrained': 'pretrained/vad_c.pth',
        'resolution': 0.02
    },
}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-w',
                       '--wav',
                       help='A single wave or any other compatible audio file')
    group.add_argument(
        '-l',
        '--wavlist',
        help=
        'A list of wave or any other compatible audio files. E.g., output of find . -type f -name *.wav > wavlist.txt'
    )
    parser.add_argument('-model', choices=list(MODELS.keys()), default='gpvf')
    parser.add_argument('-o',
                        '--output_path',
                        default=None,
                        help='Output folder to save predictions if necessary')
    parser.add_argument('-th',
                        '--threshold',
                        default=(0.5, 0.1),
                        type=float,
                        nargs="+")

    parser.add_argument('-L',
                        '--large',
                        default=None,
                        type=int,
                        help='mechanism to deal with very large wavlists. if there is no csv, it will create one with _1.csv at the end, and it will process only a 1000 files. If there is a csv already, it will process only 1000 files starting from the _n.csv , so for from file N*1000 until (N+1)*1000')
    args = parser.parse_args()

    logger.info("Passed args")
    for k, v in vars(args).items():
        logger.info(f"{k} : {str(v):<10}")
    if args.wavlist:
        wavlist = pd.read_csv(args.wavlist,
                              usecols=[0],
                              header=None,
                              names=['filename'])
        wavlist = wavlist['filename'].values.tolist()

        print("Wave list is {} files long".format(len(wavlist)))

        ### Implementing a mechanism to deal with very large wavlists
        ### argument -L or --large will check if there is already a csv in the savepath 
        ### if there is no csv, it will create one with _1.csv at the end, and it will process only a 1000 files
        ### If there is a csv already, it will process only 1000 files starting from the _n.csv , so for from file N*1000 until (N+1)*1000
        ### This mode has to be run in a loop until all files have been processed

        if args.large:
            length = args.large ### this is L
            # Check the csv files in the savepath folder and fetch the name of the most recent file
            mostrecentmodif = 0
            mostrecentfile = None
            for curfile in os.listdir(args.output_path):
                if os.path.splitext(curfile)[1]=='.csv':
                    
                    mtime = (os.path.getmtime(os.path.join(args.output_path,curfile)))
                    if mtime>mostrecentmodif:
                        mostrecentfile = curfile
                        mostrecentmodif = mtime
            if mostrecentfile is not None:
                ### The end of the file is @K.csv where K is the last file that was processed at the last run
                ### The next index at which we will end is K+L
                print("Most Recent File is {}".format(mostrecentfile))
                indL = int(mostrecentfile.split('@')[-1][:-4]) + length
            else:
                ### No file was processed so we initialise the file name to @L.csv
                mostrecentfile = os.path.join(args.output_path,"speechpreds@{}.csv".format(length))
                indL= length
                print("Creating file {} because there was no file previously processed".format(mostrecentfile))
                
            speechpredsfile = os.path.join(args.output_path,"speechpreds@{}.csv".format(indL))
            allpredsfile = os.path.join(args.output_path,"allpreds.csv".format(indL))

            wavlist = wavlist[(indL-length):indL]

            print("I will process {} files starting from file {}".format(len(wavlist),(indL-length)))
            
        else:
            speechpredsfile = os.path.join(args.output_path,"speechpredsfile@{}.csv".format(0))
            allpredsfile = os.path.join(args.output_path,"allpreds.csv".format(0))
            
        print("I will save 2 files {} and {}".format(speechpredsfile,allpredsfile))



    elif args.wav:
        wavlist = [args.wav]
    dset = OnlineLogMelDataset(wavlist, **LMS_ARGS)
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=1,
                                          num_workers=14,
                                          shuffle=False)

    print(DEVICE)
    





    model_kwargs_pack = MODELS[args.model]
    model_resolution = model_kwargs_pack['resolution']
    model = model_kwargs_pack['model'](
        outputdim=model_kwargs_pack['outputdim'],
        pretrained_file=model_kwargs_pack['pretrained']).to(DEVICE).eval()
    encoder = torch.load(model_kwargs_pack['encoder'])
    logger.trace(model)

    output_dfs = []
    threshold = tuple(args.threshold)

    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
    # Using only binary thresholding without filter
    if len(threshold) == 1:
        postprocessing_method = utils.threshold
    else:
        postprocessing_method = utils.double_threshold
    with torch.no_grad(), tqdm(total=len(dloader), leave=False,
                               unit='clip') as pbar:
        for feature, filename in dloader:
            feature = torch.as_tensor(feature).to(DEVICE)
            prediction_tag, prediction_time = model(feature)
            prediction_tag = prediction_tag.to('cpu')
            prediction_time = prediction_time.to('cpu')

            if prediction_time is not None:  # Some models do not predict timestamps

                cur_filename = filename[0]  #Remove batchsize
                thresholded_prediction = postprocessing_method(
                    prediction_time, *threshold)

                labelled_predictions = utils.decode_with_timestamps(
                    encoder, thresholded_prediction)
                pred_label_df = pd.DataFrame(
                    labelled_predictions[0],
                    columns=['event_label', 'onset', 'offset'])
                if not pred_label_df.empty:
                    pred_label_df['filename'] = cur_filename
                    pred_label_df['onset'] *= model_resolution
                    pred_label_df['offset'] *= model_resolution
                    pbar.set_postfix(labels=','.join(
                        np.unique(pred_label_df['event_label'].values)))
                    pbar.update()
                    output_dfs.append(pred_label_df)

    if len(output_dfs) > 0:
        full_prediction_df = pd.concat(output_dfs).reset_index()
        prediction_df = full_prediction_df[full_prediction_df['event_label'] ==
                                        'Speech']

        if args.output_path:
            args.output_path = Path(args.output_path)
            args.output_path.mkdir(parents=True, exist_ok=True)
            prediction_df.to_csv(speechpredsfile,
                                sep='\t',
                                index=False)
            full_prediction_df.to_csv(allpredsfile,
                                    sep='\t',
                                    index=False)
            logger.info(f"Putting results also to dir {args.output_path}")
        print(prediction_df.to_markdown(showindex=False))
    else:
        print("No Speech Found!")


if __name__ == "__main__":
    main()
