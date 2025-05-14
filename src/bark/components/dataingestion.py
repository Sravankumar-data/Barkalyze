import os
from pymongo import MongoClient
import gridfs
from src.bark.entity.config_entity import DataIngestionConfig

class DataIngestion:
  def __init__(self, config: DataIngestionConfig):
      self.config = config
 
  def download_file(self,collection_name)-> str:
      '''
      Fetch data from the uri
      '''

      try: 
        MONGO_URI = self.config.MONGO_URI
        client = MongoClient(MONGO_URI)

        db = client[collection_name]
        fs = gridfs.GridFS(db)

        # Local directory to save the retrieved dataset
        OUTPUT_DIR = self.config.local_data_file + collection_name

        # Create the root output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Iterate through all files in GridFS
        for file in fs.find():
            label = file.metadata.get("label", "unknown")
            filename = file.filename

            # Create label folder if it doesn't exist
            label_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            # Define full file path
            output_path = os.path.join(label_dir, filename)

            # Write the image to disk
            with open(output_path, 'wb') as f:
                f.write(file.read())

            print(f"[DOWNLOADED] {filename} to {label}")

      except Exception as e:
          raise e
      
  
