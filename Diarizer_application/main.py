from ASR_diar import Diarizer
from colorama import Fore, init, Style
import gc
import time
import json
from tqdm import tqdm

class InteractiveApp:

    init(autoreset=True)
    
    def __init__(self):
        print(Fore.CYAN + "Initializing diarization components...")
        self.diarizer = Diarizer()

    def run(self):

        manifest_path = "/workspace/tirocinio/AudioSample/WER_test/final_manifest.jsonl"
        with open(manifest_path, 'r', encoding='utf-8') as file:

            for line in file:
                data = json.loads(line.strip())
                
                # ADD
                file_path = data.get("audio_path")
                try:
                    self.diarizer.add_audiofile(file_path)
                    print(Fore.CYAN + "Setting configurations...")
                    self.diarizer.set_config()
                except Exception as e:
                    print(Fore.RED + f"{e}")
                print(Fore.GREEN + "File added successfully")

                # ALL
                print(Fore.CYAN + "Diarizing: ")
                try:
                    t1_start = time.perf_counter()
                    word_hyp, word_ts_hyp = self.diarizer.transcript_infer()
                    self.diarizer.diarization(word_hyp, word_ts_hyp)
                    t1_stop = time.perf_counter() 
                    print(Fore.GREEN + "-> Diarization time:", t1_stop - t1_start)
                    del word_hyp
                    del word_ts_hyp
                    gc.collect()
                except Exception as e:
                    print(Fore.RED + f"{e}")

if __name__ == "__main__":
    app = InteractiveApp()
    app.run()