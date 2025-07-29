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
        # wipe and recreate the sample_data folder
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
                dst = os.path.join(dest_class_path, img)
                shutil.copy2(src, dst)

    def main(self):
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # 1) load config & get the 100+100 dataset path
        config = ConfigurationManager()
        training_config = config.get_training_config()
        original_data_path = str(training_config.training_data)

        # 2) build 100‑per‑class sample under sample_data/
        sample_data_path = os.path.join(
            "artifacts", "data_ingestion", "kidney-ct-scan-image"
        )
        self.create_sample_dataset(
            source_dir=original_data_path,
            dest_dir=sample_data_path,
            samples_per_class=100
        )

        # 3) override config to point at sample_data/
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

        # 4) train on that sample_data/
        trainer = Training(config=new_training_config)
        trainer.get_base_model()
        trainer.train_valid_generator()
        trainer.train()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


if __name__ == '__main__':
    ModelTrainingPipeline().main()
