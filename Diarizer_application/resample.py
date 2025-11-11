"""
Prende in input il percorso assoluto del file audio e ne fa il resampling a 16K Hz e a canale mono mettendolo nella cartella 
/workspace/tirocinio/AudioSample/resampled.
"""
import os
from pydub import AudioSegment

def resample_audio(input_path):
    # Verifica se il file esiste
    if not os.path.isfile(input_path):
        print(f"Il file {input_path} non esiste.")
        return 0

    # Estrai il nome del file senza estensione
    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)

    # Percorso per il nuovo file
    output_dir = "/resampled"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.wav")

    # Carica il file audio
    audio = AudioSegment.from_file(input_path)

    # Converte in mono e imposta il frame rate a 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Esporta il nuovo file audio
    audio.export(output_path, format="wav")

    return output_path
    
