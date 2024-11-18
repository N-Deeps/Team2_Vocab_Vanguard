import time
import wandb
import os
import torch
import random
from torch.utils.data import ConcatDataset
from Architectures.ToucanTTS.ToucanTTS import ToucanTTS
from Architectures.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def build_path_to_transcript_dict_bengali_dataset1():
    root = '/audios/bengaliDataset'
    path_to_transcript = dict()
    
    with open(Path(root,'bengalidataset1.csv'),'r',encoding='utf-8') as f:
        reader = DictReader(f, delimiter=',')
        for row in reader:
            path_to_transcript[str(Path(root,row['path']))] = row['sentence']
    
    return path_to_transcript    

def build_path_to_transcript_dict_telugu_dataset1():
    root = '/audios/teluguDataset'
    path_to_transcript = dict()
    
    with open(Path(root,'telugudataset1.csv'),'r',encoding='utf-8') as f:
        reader = DictReader(f, delimiter=',')
        for row in reader:
            path_to_transcript[str(Path(root,row['path']))] = row['sentence']
    
    return path_to_transcript    


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    assert gpu_count == 1  # distributed finetuning is not supported

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_Bengali_and_Telugu_with_Accents")
    os.makedirs(save_dir, exist_ok=True)

    all_train_sets = list()
    train_samplers = list()

    # ========================
    # =    Bengali Data      =
    # ========================
    bengali_datasets = list()
    bengali_datasets.append(prepare_tts_corpus(
        transcript_dict=build_path_to_transcript_dict_bengali_dataset1(),
        corpus_dir=os.path.join(PREPROCESSING_DIR, "BengaliDataset1"),
        lang="hin",
        accent="ben"  # ADDING ACCENT ID
    ))


    # ========================
    # =    Telugu Data       =
    # ========================
    telugu_datasets = list()
    telugu_datasets.append(prepare_tts_corpus(
        transcript_dict=build_path_to_transcript_dict_telugu_dataset1(),
        corpus_dir=os.path.join(PREPROCESSING_DIR, "TeluguDataset1"),
        lang="hin",
        accent="tel"  # ADDING ACCENT ID
    ))


    # Initialize the Model
    model = ToucanTTS()

    for train_set in all_train_sets:
        train_samplers.append(torch.utils.data.RandomSampler(train_set))

    if use_wandb:
        wandb.init(
            name=f"ToucanTTS_Bengali_Telugu_with_Accents_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,
            resume="must" if wandb_resume_id is not None else None
        )

    print("Training model")
    train_loop(net=model,
               datasets=all_train_sets,
               device=device,
               save_directory=save_dir,
               batch_size=12,
               eval_lang="ben",  # YOU CAN USE "ben" OR "tel" DEPENDING ON YOUR PRIMARY EVAL LANGUAGE
               warmup_steps=500,
               lr=1e-5,
               path_to_checkpoint=os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt") if resume_checkpoint is None else resume_checkpoint,
               fine_tune=True if resume_checkpoint is None and not resume else finetune,
               resume=resume,
               steps=5000,
               use_wandb=use_wandb,
               train_samplers=train_samplers,
               gpu_count=1)
    if use_wandb:
        wandb.finish()
