import torch
import logging
import numpy as np


logger = logging.getLogger(__name__)


def load_model(path_to_checkpoint: str) -> dict:
    checkpoint = torch.load(path_to_checkpoint)
    return checkpoint


def log_metrics(metrics_d: dict, write_to_file=""):
    output_lines = []
    for module_name, metrics in metrics_d.items():
        if module_name == "full":
            continue
        logger.info("")
        logger.info("Results for module %s" % module_name)
        logger.info("")
        for target_name, target_metrics in metrics.items():
            for metric_key, metric in target_metrics.items():
                logger.info("%s for %s: %5.2f" % (metric_key, target_name, metric))
                output_line = "%s__%s %5.2f\n" % (target_name, metric_key, metric)
                output_lines.append(output_line)
        logger.info("")
    logger.info("")
    logger.info("Full accuracy: %5.2f" % metrics_d["full"]["accuracy"])
    logger.info("Full exact match: %5.2f" % metrics_d["full"]["exact_match"])
    output_lines.append("target_tensor__accuracy %5.2f\n" % metrics_d["full"]["accuracy"])
    output_lines.append("target_tensor__exact_match %5.2f\n" % metrics_d["full"]["exact_match"])

    if write_to_file:
        with open(write_to_file, "w") as outfile:
            outfile.writelines(output_lines)
    return
