from src.med_classifier.config.configuration import ConfigurationManager
from src.med_classifier.components.model_training import Training
from src.med_classifier import logger
from src.med_classifier.entity.config_entity import TrainingConfig

import os
import shutil
import random
from pathlib import Path

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def create_sample_dataset(self, source_dir, dest_dir, samples_per_class=100):
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)

        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = os.listdir(class_path)
            random.shuffle(images)
            sample_images = images[:samples_per_class]

            dest_class_path = os.path.join(dest_dir, class_name)
            os.makedirs(dest_class_path, exist_ok=True)

            for img in sample_images:
                src = os.path.join(class_path, img)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(dest_class_path, img)
                shutil.copy(src, dst)

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()

        # Create sample dataset (X logic here)
        original_data_path = training_config.training_data
        sample_data_path = os.path.join("artifacts", "sample_dataset")
        self.create_sample_dataset(original_data_path, sample_data_path, samples_per_class=100)

        # Create new TrainingConfig with updated data path
        new_training_config = TrainingConfig(
            root_dir=training_config.root_dir,
            trained_model_path=training_config.trained_model_path,
            updated_base_model_path=training_config.updated_base_model_path,
            training_data=Path(sample_data_path),
            params_epochs=training_config.params_epochs,
            params_batch_size=training_config.params_batch_size,
            params_is_augmentation=training_config.params_is_augmentation,
            params_image_size=training_config.params_image_size
        )

        # Train on sampled data
        training = Training(config=new_training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
