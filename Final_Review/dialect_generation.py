import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor,energy_variance_scale=1.0,silent=True, pause_duration_scaling_factor=1.0 )
    del tts


def dialect_generation(version, model_id="Meta", device="cpu", speaker_reference=None):
    os.makedirs("audios/generated/", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""थोड़ा थोड़ा हिंदी आता है मेरे को । """],
               filename=f"audios/generated/{version}_1.wav",
               device=device,
               language="eng",
               speaker_reference=speaker_reference,
               duration_scaling_factor=1.2)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")

    # spkr_ref_file = "./audios/telugu_ref/"
    # merged_speaker_ref = [spkr_ref_file + ref for ref in os.listdir(spkr_ref_file)]
    
    merged_speaker_ref = None
    dialect_generation(version="GT_Telugu_hin_1",
              model_id="Meta",
              device=device,
              speaker_reference=merged_speaker_ref)
    print("done")
    
