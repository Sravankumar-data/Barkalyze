import os
import shutil
import random
import subprocess
from src.bark.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.train_ratio = 0.8
    def download_file(self):
        """
        Pull data from DVC remote by running 'dvc pull',
        then split into train/test folders locally.
        """
        try:
        # Step 1: Run 'dvc pull' to sync data from remote storage
            try:
                print("[INFO] Pulling data from DVC remote...")
                result = subprocess.run(["dvc", "pull", os.path.join(self.config.root_dir, "emotion_dataset.dvc")], capture_output=True, text=True, check=True)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] dvc pull failed: {e.stderr}")
                raise e
            
            data_dir = os.path.join(self.config.root_dir,"emotion_dataset")
            train_dir = os.path.join(self.config.local_data_file, "emotion_dataset_train")
            test_dir = os.path.join(self.config.local_data_file, "emotion_dataset_test")

            # Remove old train/test folders if they exist
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Step 2: Split data into train/test folders by emotion
            for emotion_folder in os.listdir(data_dir):
                emotion_path = os.path.join(data_dir, emotion_folder)
                if not os.path.isdir(emotion_path):
                    continue

                files = os.listdir(emotion_path)
                # random.shuffle(files)

                split_idx = int(len(files) * self.train_ratio)
                train_files = files[:split_idx]
                test_files = files[split_idx:]

                train_emotion_dir = os.path.join(train_dir, emotion_folder)
                test_emotion_dir = os.path.join(test_dir, emotion_folder)
                os.makedirs(train_emotion_dir, exist_ok=True)
                os.makedirs(test_emotion_dir, exist_ok=True)

                for f in train_files:
                    shutil.copy2(os.path.join(emotion_path, f), os.path.join(train_emotion_dir, f))

                for f in test_files:
                    shutil.copy2(os.path.join(emotion_path, f), os.path.join(test_emotion_dir, f))

                print(f"[SPLIT] {emotion_folder}: {len(train_files)} train, {len(test_files)} test")
        except Exception as e:
          raise e
      
  
