import logging
from dataclasses import dataclass, field
from config.GetPath import paths
from utils.config import Config
from services.data_module import SolarPanelDataModule
from services.model_factory import ModelFactory
from services.trainer import Trainer, AdvancedTrainer
from services.evaluator import Evaluator, TTAEvaluator

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class RunConfig:
    models: list[str] = field(default_factory = lambda: ["efficientnetb4", "resnet50", "vit"])
    use_advanced_trainer: bool = True # Use AdvancedTrainer (Mixup + Label Smoothing + LLRD)

class TrainingPipeline:

    def __init__(self, run_cfg: RunConfig):
        self.run_cfg = run_cfg
        self.dm = SolarPanelDataModule(
            img_size = Config.IMG_SIZE,
            batch_size = Config.BATCH_SIZE,
            seed = Config.SEED
        )
        self.evaluator = Evaluator(device = Config.DEVICE, class_names = Config.CLASS_NAMES)
        logger.info("TrainingPipeline initialized | models = %s", run_cfg.models)

    def run(self) -> None:
        logger.info("========== Pipeline START ==========")
        Config.seed_everything()
        Config.summary()
        self._prepare_data()
        loaders = self.dm.get_loaders()
        class_counts = Config.get_class_counts(paths.data)
        logger.info("Class counts for loss weighting: %s", dict(zip(Config.CLASS_NAMES, class_counts)))

        for model_name in self.run_cfg.models:
            self._train_model(model_name, loaders, class_counts)

        logger.info("========== Pipeline DONE ==========")

    def _prepare_data(self) -> None:
        logger.info("---------- Data Preparation ----------")
        self.dm.setup()

    def _train_model(self, model_name: str, loaders: dict, class_counts: list[int]) -> None:
        logger.info("========== Model: %s ==========", model_name.upper())
        logger.info("[%s] Phase 1 — Head warm-up (Backbone frozen)", model_name)
        model = ModelFactory.build(model_name, freeze_backbone = True)
        trainer = self._make_trainer(model, model_name, loaders, class_counts)
        trainer.train(
            num_epochs = Config.NUM_EPOCHS_PHASE1,
            lr = Config.LR_PHASE1,
            phase_label = "Phase 1 — Head Warm-up"
        )
        logger.info("[%s] Evaluating after Phase 1...", model_name)
        y_true, y_pred, _ = self.evaluator.predict(model, loaders["val"])
        self.evaluator.print_report(model_name, y_true, y_pred, split = "Val Phase 1")
        logger.info("[%s] Phase 2 — Full fine-tune (backbone unfrozen)", model_name)
        trainer.unfreeze_all()
        if self.run_cfg.use_advanced_trainer:
            adv_trainer = AdvancedTrainer(
                model, model_name, loaders, class_counts,
                label_smoothing = Config.LABEL_SMOOTHING,
                mixup_alpha = Config.MIXUP_ALPHA
            )
            adv_trainer.train_with_llrd(
                num_epochs = Config.LLRD_EPOCHS,
                base_lr = Config.LLRD_BASE_LR,
                decay_factor = Config.LLRD_DECAY
            )
            for k in trainer.history:
                trainer.history[k].extend(adv_trainer.history[k])
        else:
            trainer.train(
                num_epochs = Config.NUM_EPOCHS_PHASE2,
                lr = Config.LR_PHASE2,
                phase_label = "Phase 2 — Full Fine-tune"
            )
        logger.info("[%s] Evaluating after Phase 2 (Val)...", model_name)
        y_true, y_pred, _ = self.evaluator.predict(model, loaders["val"])
        self.evaluator.print_report(model_name, y_true, y_pred, split = "Val Phase 2")
        logger.info("[%s] Evaluating on Test set...", model_name)
        y_true, y_pred, _ = self.evaluator.predict(model, loaders["test"])
        self.evaluator.print_report(model_name, y_true, y_pred, split = "Test")
        logger.info("[%s] Running TTA evaluation...", model_name)
        test_indices = self.dm.test_dataset.indices  # type: ignore[union-attr]
        tta_eval = TTAEvaluator(
            data_dir = paths.data,
            test_indices = test_indices,
            device = Config.DEVICE,
            class_names = Config.CLASS_NAMES,
            batch_size = Config.BATCH_SIZE,
            img_size = Config.IMG_SIZE
        )
        y_true_tta, y_pred_tta = tta_eval.predict(model)
        self.evaluator.print_report(model_name, y_true_tta, y_pred_tta, split = "Test TTA")
        trainer.save()
        logger.info("[%s] Model saved.", model_name)
        trainer.plot_history()

    def _make_trainer(self, model, model_name: str, loaders: dict, class_counts: list[int]) -> Trainer:
        return Trainer(
            model = model,
            model_name = model_name,
            loaders = loaders,
            class_counts = class_counts,
            device = Config.DEVICE,
            models_dir = paths.models,
            patience = 5
        )

if __name__ == "__main__":
    run_cfg = RunConfig()
    pipeline = TrainingPipeline(run_cfg)
    pipeline.run()