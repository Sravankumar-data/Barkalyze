import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
from src.bark.entity.config_entity import PrepareBaseModelConfig
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = MobileNetV2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till):
        model.trainable = True
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
                
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        output = Dense(classes, activation='softmax')(x)  
        full_model = Model(inputs=model.input, outputs=output)

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=30,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path, include_optimizer=False)