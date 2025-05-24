from src.bark.config.configuration import ConfigurationManager
from src.bark.components.datadrift import DataDrift
from src.bark import logger



STAGE_NAME = "Data Drift stage"

class DataDriftPipeline:
    def __init__(self):
        pass

    def main(self):
      config = ConfigurationManager()
      data_drift_config = config.get_data_drift_config()
      data_drift = DataDrift(config=data_drift_config)
      data_drift.generate_csv_from_image_folder(data_drift_config.reference_data_path, data_drift_config.output_train_path)
      data_drift.generate_csv_from_image_folder(data_drift_config.current_data_path, data_drift_config.output_test_path)
      data_drift.generate_report()


if __name__ == '__main__':
    try:

        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataDriftPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
