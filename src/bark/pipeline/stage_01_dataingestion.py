from src.bark.config.configuration import ConfigurationManager
from src.bark.components.dataingestion import DataIngestion
from src.bark import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
      config = ConfigurationManager()
      data_ingestion_config = config.get_data_ingestion_config()
      data_ingestion = DataIngestion(config=data_ingestion_config)
      data_ingestion.download_file("emotion_dataset_test")
      data_ingestion.download_file("emotion_dataset_train")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
