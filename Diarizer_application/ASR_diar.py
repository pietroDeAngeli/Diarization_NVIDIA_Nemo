import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
import os
import glob
import json
from librosa import get_duration, load
from omegaconf import OmegaConf
import resample
from typing import Dict, List 
import gc
import conversion

class Diarizer:

    # Define models name
    pretrained_speaker_model = 'models/titanet-l.nemo'
    pretrained_vad_model = 'models/vad_multilingual_marblenet.nemo'
    
    def __init__(self):
        """
        Initializes an instance of the AudioProcessor class.

        Attributes:
        - self.AUDIO_FILEPATH: Path to the audio file (initially empty).
        - self.AUDIO_FILENAME: Name of the audio file (initially empty).
        - self.duration: Duration of the audio file (initially 0).
        - self.data_dir: Path to the 'data' directory, created in the current working directory.
        - self.cfg: YAML configuration for diarization, downloaded if not already present.

        Functionality:
        1. Creates the 'data' directory in the current working directory if it doesn't exist.
        2. Downloads and loads a YAML configuration file for diarization if not already available.
        3. Sets the parameters for diarization, VAD, and speaker embeddings.
        4. Restores a pre-trained ASR model and creates an instance of the diarizer using the loaded configuration.
        """

        self.AUDIO_FILEPATH = ""
        self.AUDIO_FILENAME = ""
        self.duration = 0

        # Creating default directories
        ROOT = os.getcwd()
        self.data_dir = os.path.join(ROOT,'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Set default config
        DOMAIN_TYPE = "general"
        CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"

        if not os.path.exists(os.path.join(self.data_dir,CONFIG_FILE_NAME)):
            import wget
            CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
            CONFIG = wget.download(CONFIG_URL, self.data_dir)
        else:
            CONFIG = os.path.join(self.data_dir,CONFIG_FILE_NAME)
        self.cfg = OmegaConf.load(CONFIG)

        # Update diarizer configs
        self.cfg.verbose = False
        #self.cfg.batch_size = 64

        self.cfg.diarizer.manifest_filepath =  os.path.join(self.data_dir,'input_manifest.json')
        self.cfg.diarizer.out_dir = self.data_dir

        # Speaker identification configs
        self.cfg.diarizer.speaker_embeddings.model_path = self.pretrained_speaker_model
        self.cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.2] 
        self.cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.6] 
        self.cfg.diarizer.speaker_embeddings.parameters.multiscale_weights = [1]
        self.cfg.diarizer.speaker_embeddings.parameters.save_embeddings = False

        # Clustering
        self.cfg.diarizer.clustering.parameters.oracle_num_speakers=False
        self.cfg.diarizer.clustering.parameters.embeddings_per_chunk = 1200

        # VAD configs
        self.cfg.diarizer.vad.model_path = self.pretrained_vad_model
        self.cfg.diarizer.oracle_vad = False
        self.cfg.diarizer.vad.parameters.onset = 0.8
        self.cfg.diarizer.vad.parameters.offset = 0.6
        self.cfg.diarizer.vad.parameters.pad_offset = -0.05

        # Restore ASR model
        self.asr_model = nemo_asr.models.ASRModel.restore_from("models/stt_multilingual_fastconformer_hybrid_large_pc_timestamps.nemo")
        
        # Restore diarizer model
        self.asr_diar_offline = OfflineDiarWithASR(self.cfg)


    def add_audiofile(self, audio_filepath):
        """
        Loads an audio file and prepares it for processing.

        Parameters:
        - audio_filepath (str): Path to the input audio file.

        Functionality:
        1. Stores the path and filename (without extension) of the audio file.
        2. Loads the audio signal and determines its duration.
        3. Checks the sampling rate:
        - If it's not 16 kHz, the audio is resampled to 16 kHz.
        - Updates the filepath and filename accordingly.
        """
        # Initialize directories and filenames
        self.AUDIO_FILEPATH = audio_filepath
        self.AUDIO_FILENAME = os.path.splitext(os.path.basename(self.AUDIO_FILEPATH))[0]
        self.duration = 0

        # Audio manipulation
        signal, sr = load(self.AUDIO_FILEPATH, sr=None)
        self.duration = get_duration(y=signal, sr=sr)

        # Resampling if necessary
        if sr != 16000:
            self.AUDIO_FILEPATH = resample.resample_audio(self.AUDIO_FILEPATH)
            self.AUDIO_FILENAME = os.path.splitext(os.path.basename(self.AUDIO_FILEPATH))[0]

    def set_config(self):
        """
        Creates a JSON manifest file required for diarization.

        Functionality:
        1. Prepares metadata for the input audio, including:
        - File path
        - Offset and duration
        - Placeholder label and text
        - Speaker and RTTM/UEM info (set to None for inference)
        2. Writes the metadata to 'input_manifest.json' inside the data directory.
        """
        # Create a manifest file for input
        meta = {
            'audio_filepath': self.AUDIO_FILEPATH,
            'offset': 0,
            'duration': self.duration,
            'label': 'infer',
            'text': '-',
            'num_speakers': None,
            'rttm_filepath': None,
            'uem_filepath' : None
        }
        with open(os.path.join(self.data_dir,'input_manifest.json'),'w') as fp:
            json.dump(meta,fp)
            fp.write('\n')

    def transcript_infer(self):
        """
        Performs transcription inference on the input audio file.

        Functionality:
        1. Uses the ASR model to transcribe the audio and obtain timestamped hypotheses.
        2. Parses the transcription into:
        - word_hyp: a dictionary mapping filename to a list of recognized words or characters.
        - word_ts_hyp: a dictionary mapping filename to a list of [start, end] timestamps for each word.
        3. Computes absolute time offsets using the model’s time stride.

        Returns:
        - word_hyp (Dict[str, List[str]]): Transcribed words or characters.
        - word_ts_hyp (Dict[str, List[List[int]]]): Corresponding time intervals in seconds.
        """
        # Transcription
        hypotheses = self.asr_model.transcribe([self.AUDIO_FILEPATH], return_hypotheses=True, batch_size=2)

        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

        # Parsing transcription
        timestamp_dict = hypotheses[0].timestep 
        time_stride = 8 * self.asr_model.cfg.preprocessor.window_stride

        word_hyp: Dict[str, List[str]] = {}
        word_ts_hyp: Dict[str, List[List[int]]] = {}

        word_hyp[self.AUDIO_FILENAME] = list()
        word_ts_hyp[self.AUDIO_FILENAME] = list()

        for stamp in timestamp_dict['word']:
            word_hyp[self.AUDIO_FILENAME].append(stamp['char'] if 'char' in stamp else stamp['word'])
            word_ts_hyp[self.AUDIO_FILENAME].append([stamp['start_offset'] * time_stride, stamp['end_offset'] * time_stride])

        return word_hyp, word_ts_hyp

    def diarization(self, word_hyp, word_ts_hyp):
        """
        Performs speaker diarization using the loaded configuration.

        Parameters:
        - word_hyp (dict): Dictionary of transcribed words/characters.
        - word_ts_hyp (dict): Dictionary of their corresponding timestamps.

        Functionality:
        1. Runs offline diarization and retrieves speaker-labeled segments.
        2. Aligns the speaker labels with transcribed words.
        3. Clears memory used by diarization output.
        """
        diar_hyp, _ = self.asr_diar_offline.run_diarization(self.cfg, None)

        self.asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
        
        del diar_hyp
        gc.collect()
            
    def convertJSON(self):
        """
        Converts diarization output to a final JSON format and cleans intermediate files.

        Functionality:
        1. Converts the diarization result (stored in JSON format) to the final output format.
        2. Deletes intermediate prediction files generated during diarization.
        3. Handles potential file deletion errors gracefully.
        """
        input_file = f"/workspace/tirocinio/Diar_application/data/pred_rttms/{self.AUDIO_FILENAME}.json"
        output_file = f"/workspace/tirocinio/Diar_application/data/output/{self.AUDIO_FILENAME}.json"
        conversion.convert_json_from_file(input_file, output_file)

        # Cleaning intermediate files 
        pattern = f"/workspace/tirocinio/Diar_application/data/pred_rttms/{self.AUDIO_FILENAME}.*"
        files_to_delete = glob.glob(pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

        patterns = [
            f"data/pred_rttms/{self.AUDIO_FILENAME}.*",
            f"data/pred_rttms/{self.AUDIO_FILENAME}_gecko.*"
        ]
        
        for pattern in patterns:
            files_to_delete = glob.glob(pattern)
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")