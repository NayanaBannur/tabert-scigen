import argparse
import gc
import glob
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers import (
    AdamW,
    BartConfig,
    get_linear_schedule_with_warmup,
    BartForConditionalGeneration,
    BartTokenizer
)

from callbacks import get_checkpoint_callback, get_early_stopping_callback
from table_bert import TableBertConfig, TableBertModel
from utils.scigen_utils import convert_text, eval_sacre_bleu, eval_mover_score
from dataloader import TableDataset

gc.collect()
torch.cuda.empty_cache()

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

ENCODER_PATH = './tabert_base_k3/'


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Seq2SeqTableBertModel(pl.LightningModule):

    val_metric = "mover"

    def __init__(self, hparams: argparse.Namespace, num_labels=None, **config_kwargs):
        """Initialize a model."""

        super().__init__()

        self.hparams = hparams

        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.config_encoder = TableBertConfig.from_file(f'{ENCODER_PATH}/tb_config.json')

        self.config_decoder = BartConfig.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
            is_encoder_decoder=False,
            **config_kwargs
        )

        self.encoder = TableBertModel.from_pretrained(
            f'{ENCODER_PATH}/model.bin',
        )

        self.decoder = BartForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config_decoder,
            cache_dir=cache_dir,
        )

        self.tokenizer_encoder = self.encoder.tokenizer

        self.tokenizer_decoder = BartTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=cache_dir,
        )

        self.step_count = 0
        self.test_type = self.hparams.test_type
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_target_length=self.hparams.max_target_length,
        )
        self.count_valid_epoch = 0

    def is_logger(self):
        return True

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        optimizer_grouped_parameters = []
        no_decay = ["bias", "LayerNorm.weight"]
        for model in [self.encoder, self.decoder]:
            optimizer_grouped_parameters.extend([
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ])
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        tqdm_dict = {"loss": "{:.3f}".format(avg_training_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def forward(self, tensor_dict=None, decoder_input_ids=None, labels=None):
        context_encoding, schema_encoding = self.forward(**tensor_dict)
        tensor_dict['context_token_mask'] = tensor_dict['context_token_mask'][:, 0, :]
        tensor_dict['column_mask'] = tensor_dict['table_mask'][:, 0, :]

        encoding = torch.cat([context_encoding, schema_encoding], dim=1)
        mask = torch.cat([tensor_dict['context_token_mask'], tensor_dict['column_mask']], dim=1)

        return self.decoder.forward(input_ids=encoding, attention_mask=mask, decoder_input_ids=decoder_input_ids,
                                    labels=labels)

    def _step(self, batch):
        pad_token_id = self.tokenizer_decoder.pad_token_id
        tensor_dict, y = batch["tensor_dict"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone()
        labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(tensor_dict=tensor_dict, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def _generation_common(self, batch):
        tensor_dict, y = batch["tensor_dict"], batch["target_ids"]
        context_encoding, schema_encoding = self.forward(**tensor_dict)
        tensor_dict['context_token_mask'] = tensor_dict['context_token_mask'][:, 0, :]
        tensor_dict['column_mask'] = tensor_dict['table_mask'][:, 0, :]

        encoding = torch.cat([context_encoding, schema_encoding], dim=1)
        mask = torch.cat([tensor_dict['context_token_mask'], tensor_dict['column_mask']], dim=1)
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        generated_ids = self.decoder.generate(
            input_ids=encoding,
            attention_mask=mask,
            num_beams=5,
            max_length=512,
            length_penalty=5.0,
            early_stopping=True,
            use_cache=True,
        )
        preds = [
            self.tokenizer_decoder.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [self.tokenizer_decoder.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        loss = self._step(batch)
        return {"val_loss": loss, "preds": preds, "target": target}

    def validation_step(self, batch):
        return self._generation_common(batch)

    def test_step(self, batch):
        return self._generation_common(batch)

    def test_epoch_end(self, outputs):
        if "preds" in outputs[0]:
            output_test_predictions_file = os.path.join(self.hparams.output_dir,
                                                        "test_predictions_" + self.hparams.test_type + "_" +
                                                        str(self.count_valid_epoch) + ".txt")
            output_test_targets_file = os.path.join(self.hparams.output_dir,
                                                    "test_targets_" + self.hparams.test_type + "_" +
                                                    str(self.count_valid_epoch) + ".txt")
            # write predictions and targets for later rouge evaluation.
            with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
                for output_batch in outputs:
                    p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
                    t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])
                p_writer.close()
                t_writer.close()

            # bleu_info = eval_bleu_sents(output_test_targets_file, output_test_predictions_file)
            bleu_info = eval_sacre_bleu(output_test_targets_file, output_test_predictions_file)
            # bleu_info = eval_bleu(output_test_targets_file, output_test_predictions_file)
            moverScore = eval_mover_score(output_test_targets_file, output_test_predictions_file)

            logger.info("valid epoch: %s", self.count_valid_epoch)
            logger.info("%s bleu_info: %s", self.count_valid_epoch, bleu_info)
            logger.info("%s mover score: %s", self.count_valid_epoch, moverScore)

            output_test_metrics_file = os.path.join(self.hparams.output_dir,
                                                    "test_metrics_" + self.hparams.test_type + "_" +
                                                    str(self.count_valid_epoch) + ".txt")
            with open(output_test_metrics_file, "w") as writer:
                writer.write(str(bleu_info) + "\n" + str(moverScore) + "\n")
                writer.close()

            self.count_valid_epoch += 1

        else:
            logger.info('not in')

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs, prefix="val"):
        self.step_count += 1

        if "preds" in outputs[0]:
            output_test_predictions_file = os.path.join(self.hparams.output_dir, "validation_predictions_" +
                                                        str(self.count_valid_epoch) + ".txt")
            output_test_targets_file = os.path.join(self.hparams.output_dir, "validation_targets_" +
                                                    str(self.count_valid_epoch) + ".txt")
            # write predictions and targets for later rouge evaluation.
            with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
                for output_batch in outputs:
                    p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
                    t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])
                p_writer.close()
                t_writer.close()

            # bleu_info = eval_bleu_sents(output_test_targets_file, output_test_predictions_file)
            if self.count_valid_epoch >= 0:
                bleu_info = eval_sacre_bleu(output_test_targets_file, output_test_predictions_file)
                moverScore = eval_mover_score(output_test_targets_file, output_test_predictions_file)
                output_val_metrics_file = os.path.join(self.hparams.output_dir, "validation_metrics_" +
                                                       str(self.count_valid_epoch) + ".txt")
                with open(output_val_metrics_file, "w") as writer:
                    writer.write(str(bleu_info) + "\n" + str(moverScore) + "\n")
                    writer.close()
            else:
                bleu_info = 0
                moverScore = [0, 0]

            metrics = {}
            metrics["{}_avg_bleu".format(prefix)] = bleu_info
            metrics["{}_mover_mean1".format(prefix)] = moverScore[0]
            metrics["{}_mover_median1".format(prefix)] = moverScore[1]
            metrics["step_count"] = self.step_count

            logger.info("valid epoch: %s", self.count_valid_epoch)
            logger.info("%s bleu_info: %s", self.count_valid_epoch, bleu_info)
            logger.info("%s mover score: %s", self.count_valid_epoch, moverScore)

            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

            mover_tensor: torch.FloatTensor = torch.tensor(moverScore[0]).type_as(avg_loss)

            self.count_valid_epoch += 1

        else:
            logger.info('not in')
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        output_val_info_file = os.path.join(self.hparams.output_dir, "validation_info_" +
                                            str(self.count_valid_epoch - 1) + ".txt")
        with open(output_val_info_file, "w") as writer:
            writer.write(str(avg_loss) + "\n")
            writer.write(str(mover_tensor) + "\n")
            writer.close()

        return {"avg_val_loss": avg_loss, "log": metrics, "{}_mover".format(prefix): mover_tensor}

    def train_epoch_end(self, outputs, prefix='train'):
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        output_train_info_file = os.path.join(self.hparams.output_dir, "train_info_" +
                                              str(self.count_valid_epoch) + ".txt")
        with open(output_train_info_file, "w") as writer:
            writer.write(str(avg_loss) + "\n")
            writer.close()

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = TableDataset(self.tokenizer_encoder, self.tokenizer_decoder, type_path=type_path, **self.dataset_kwargs)
        logger.info('loading %s dataloader...', type_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                                collate_fn=dataset.collate_fn)
        logger.info('done')
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(f'test_{self.hparams.test_type}', batch_size=self.hparams.test_batch_size)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pretrained model or model identifier from huggingface.co/models")
        parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_train_epochs", default=3, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--test_batch_size", default=512, type=int)
        parser.add_argument("--max_source_length", default=384, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                 "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--max_target_length", default=512, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                 "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--data_dir", default=None, type=str, required=True,
                            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.")
        parser.add_argument("--early_stopping_patience", type=int, default=-1, required=False,
                            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.")
        parser.add_argument("--checkpoint", default=None, type=str, help="The checkpoint to initialize model")
        parser.add_argument("--checkpoint_model", default=None, type=str,
                            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.")
        parser.add_argument("--output_dir", default=None, type=str, required=False,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--test_type", default=None, type=str, required=True, help="The test file.")

        parser.add_argument("--fp16", action="store_true",
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument("--fp16_opt_level", type=str, default="O1",
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html", )
        parser.add_argument("--n_gpu", type=int, default=1)
        parser.add_argument("--n_tpu_cores", type=int, default=0)
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        return parser


class LoggingCallback(pl.Callback):
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log results
            epoch = metrics['epoch']
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, f"info_{epoch}.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        val = metrics[key]
                        if isinstance(val, torch.Tensor):
                            val = val.cpu().detach().numpy()
                        else:
                            val = str(val)
                        writer.write("{} = {}".format(key, val))
                        writer.write('\n')
                        logger.info("{} = {}".format(key, str(metrics[key])))
            writer.close()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}".format(key, str(metrics[key])))
                        writer.write("{} = {}".format(key, str(metrics[key])))


def generic_train(model: Seq2SeqTableBertModel, args: argparse.Namespace,
                  early_stopping_callback=False, checkpoint_callback=None,
                  ):
    # init model
    set_seed(args)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not checkpoint_callback:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=-1
        )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=early_stopping_callback,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
        log_save_interval=1,
        num_sanity_val_steps=4,
        reload_dataloaders_every_epoch=True
    )

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_tpu_cores > 0:
        global xm

        train_params["num_tpu_cores"] = args.n_tpu_cores
        train_params["gpus"] = 0

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(**train_params)

    if args.do_train:
        trainer.fit(model)

    return trainer


def main(args):
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{time.strftime('%Y%m%d_%H%M%S')}", )
        os.makedirs(args.output_dir)
    model = Seq2SeqTableBertModel(args)
    if args.checkpoint_model:
        model = model.load_from_checkpoint(args.checkpoint_model)
        logger.info("args.data_dir: %s", args.data_dir)
        model.dataset_kwargs: dict = dict(
            data_dir=args.data_dir,
            max_target_length=args.max_target_length
        )
        model.hparams = args

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    trainer = generic_train(model, args,
                            checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
                            early_stopping_callback=es_callback)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        if args.checkpoint_model:
            trainer.test(model)
        else:
            checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
            if checkpoints:
                print('Loading weights from {}'.format(checkpoints[-1]))
                model = model.load_from_checkpoint(checkpoints[-1])
                model.dataset_kwargs: dict = dict(
                    data_dir=args.data_dir,
                    max_target_length=args.max_target_length
                )
                model.hparams = args

            trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Seq2SeqTableBertModel.add_args(parser)
    args = parser.parse_args()
    main(args)
