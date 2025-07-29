from src.med_classifier import logger
from src.med_classifier.pipeline.step_1_data_ingest import DataIngestionTrainingPipeline
from src.med_classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.med_classifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from src.med_classifier.entity.config_entity import TrainingConfig
from src.med_classifier.pipeline.stage_04_model_evaluation import EvaluationPipeline
from pathlib import Path
import os
import random
import shutil


### STEP 1: Data Ingestion
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


### STEP 2: Prepare Base Model
STAGE_NAME = "Prepare base model"
try: 
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


### STEP 3: Create 100-100 Sample Dataset
def create_sample_dataset(original_data_path, sample_data_path, samples_per_class=100):
    classes = ["Normal", "Tumor"]
    os.makedirs(sample_data_path, exist_ok=True)

    for cls in classes:
        src_dir = os.path.join(original_data_path, cls)
        dst_dir = os.path.join(sample_data_path, cls)
        os.makedirs(dst_dir, exist_ok=True)

        all_images = os.listdir(src_dir)
        sampled_images = random.sample(all_images, min(samples_per_class, len(all_images)))

        for img in sampled_images:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))

# Location of full dataset
original_data_path = os.path.join(
    "artifacts", 
    "data_ingestion", 
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone", 
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
)

# Location to create sample set
sample_data_path = os.path.join("artifacts", "data_ingestion", "sample_data")

# Create sample dataset
create_sample_dataset(original_data_path, sample_data_path, samples_per_class=100)


### STEP 4: Inject Sample Dataset into Training
STAGE_NAME = "Training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    # Load training config and inject new dataset path
    from src.med_classifier.config.configuration import ConfigurationManager
    from src.med_classifier.components.model_training import Training

    config = ConfigurationManager()
    training_config = config.get_training_config()
    
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

    # Run training manually (bypassing pipeline)
    training = Training(config=new_training_config)
    training.get_base_model() 

    training.train_valid_generator()
    training.train()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e