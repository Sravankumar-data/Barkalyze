import os
import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount
from src.bark.entity.config_entity import DataDriftConfig

class DataDrift:
    def __init__(self, config: DataDriftConfig):
        self.config = config
    def generate_csv_from_image_folder(self,image_root_dir: str, output_csv_path: str):
        """
        Converts a directory structure like:
        - emotion_dataset/happy/image1.jpg
        - emotion_dataset/sad/image2.jpg
        into a CSV:
        filepath,label
        happy/image1.jpg,happy
        sad/image2.jpg,sad
        """

        records = []

        for label_folder in os.listdir(image_root_dir):
            label_path = os.path.join(image_root_dir, label_folder)
            if not os.path.isdir(label_path):
                continue

            for img_file in os.listdir(label_path):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    rel_path = os.path.join(label_folder, img_file)
                    records.append({
                        "filepath": rel_path.replace("\\", "/"),
                        "label": label_folder
                    })

        df = pd.DataFrame(records)
        df.to_csv(output_csv_path, index=False)
        print(f"[INFO] CSV saved to {output_csv_path}")

    def load_data(self, data_path: str):
        return pd.read_csv(data_path)
    
    def generate_report(self):
        reference_df = self.load_data(self.config.output_train_path)
        current_df = self.load_data(self.config.output_test_path)

        report = Report(metrics=[
            DriftedColumnsCount()
        ])
        my_evel = report.run(reference_data=reference_df, current_data=current_df)
        
        # Save the report
        os.makedirs(os.path.dirname(self.config.report_dir), exist_ok=True)
        my_evel.save_html(self.config.report_path)
        print(f"[INFO] Drift report saved at {self.config.report_path}")

      
  
