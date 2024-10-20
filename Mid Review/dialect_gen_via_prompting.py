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
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""यह अभिषेक, ओइंड्रिला, दीपिका, दृष्टि और सार्थक द्वारा बनाया गया एक प्रोजेक्ट है. अपनी प्रतिक्रिया देने के लिए आपका हार्दिक स्वागत है ."""],
               filename=f"audios/{version}_hindi.wav",
               device=device,
               language="hin",
               speaker_reference=speaker_reference,
               duration_scaling_factor=1.2)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")

    merged_speaker_ref = ["audios/bihar_spkr_ref/" + ref for ref in os.listdir("audios/bihar_spkr_ref/")]
   
    dialect_generation(version="bihari_accent_1",
              model_id="Meta",
              device=device,
              speaker_reference=merged_speaker_ref)
