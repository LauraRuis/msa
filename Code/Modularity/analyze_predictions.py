import numpy as np
import json
from xlwt import Workbook
from typing import List, Dict
from collections import Counter, defaultdict


def read_predictions(file: str):
    with open(file, 'r') as infile:
        data = json.load(infile)
    return data


def analyze_predictions(data: List[dict], module_target_key: str):
    exact_matches = []
    all_accuracies = []
    error_analysis = {
        "input_length": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
        "module_input_length": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
        "module_target_length": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
        "verb": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
        "adverb": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
        "type_adverb": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
        "target_object_size": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
        "correct_verb": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
    }
    for predicted_example in data:
        example_information = {}  # TODO: hier gebleven!
        example_information["accuracy"] = predicted_example["accuracy"]
        example_information["exact_match"] = predicted_example["exact_match"]
        all_accuracies.append(predicted_example["accuracy"])
        exact_matches.append(predicted_example["exact_match"])
        example_information["input_length"] = len(predicted_example["input"])
        example_information["module_input_length"] = len(predicted_example["module_input"])
        example_information["module_target_length"] = len(predicted_example["module_targets"])
        example_information["verb"] = predicted_example["verb"]
        example_information["adverb"] = predicted_example["adverb"]
        example_information["type_adverb"] = predicted_example["type_adverb"]
        example_information["target_object_size"] = predicted_example["situation"]["target_object"]["object"]["size"]
        example_information["prediction"] = predicted_example[module_target_key + "_prediction"]
        if len(predicted_example["module_targets"]) > 0:
            predicted_verbs = set(example_information["prediction"][:-1])
            example_information["predicted_verbs"] = ' - '.join(predicted_verbs)
        else:
            example_information["predicted_verbs"] = example_information["verb"]
        example_information["correct_verb"] = example_information["predicted_verbs"] == example_information["verb"]
        for key in error_analysis.keys():
            error_analysis[key][example_information[key]]["accuracy"].append(predicted_example["accuracy"])
            error_analysis[key][example_information[key]]["exact_match"].append(predicted_example["exact_match"])
    return error_analysis, exact_matches, all_accuracies


def write_error_analysis(error_analysis: Dict[str, Counter], exact_matches: List[bool], all_accuracies: List[float],
                         output_file: str):
    workbook = Workbook()
    # Write the information to a file and make plots
    with open(output_file, 'w') as outfile:
        outfile.write("Error Analysis\n\n")
        outfile.write(" Mean accuracy: {}\n".format(np.mean(np.array(all_accuracies))))
        exact_matches_counter = Counter(exact_matches)
        outfile.write(" Num. exact matches: {}\n".format(exact_matches_counter[True]))
        outfile.write(" Num not exact matches: {}\n".format(exact_matches_counter[False]))
        outfile.write(" Exact match percentage: {}\n\n".format(
            exact_matches_counter[True] / (exact_matches_counter[True] + exact_matches_counter[False])))
        for key, values in error_analysis.items():
            sheet = workbook.add_sheet(key)
            sheet.write(0, 0, key)
            sheet.write(0, 1, "Num examples")
            sheet.write(0, 2, "Mean accuracy")
            sheet.write(0, 3, "Std. accuracy")
            sheet.write(0, 4, "Exact Match")
            sheet.write(0, 5, "Not Exact Match")
            sheet.write(0, 6, "Exact Match Percentage")
            outfile.write("\nDimension {}\n\n".format(key))
            means = {}
            standard_deviations = {}
            num_examples = {}
            exact_match_distributions = {}
            exact_match_relative_distributions = {}
            for i, (item_key, item_values) in enumerate(values.items()):
                outfile.write("  {}:{}\n\n".format(key, item_key))
                accuracies = np.array(item_values["accuracy"])
                mean_accuracy = np.mean(accuracies)
                means[item_key] = mean_accuracy
                num_examples[item_key] = len(item_values["accuracy"])
                standard_deviation = np.std(accuracies)
                standard_deviations[item_key] = standard_deviation
                exact_match_distribution = Counter(item_values["exact_match"])
                exact_match_distributions[item_key] = exact_match_distribution
                exact_match_relative_distributions[item_key] = exact_match_distribution[True] / (
                        exact_match_distribution[False] + exact_match_distribution[True])
                outfile.write("    Num. examples: {}\n".format(len(item_values["accuracy"])))
                outfile.write("    Mean accuracy: {}\n".format(mean_accuracy))
                outfile.write("    Min. accuracy: {}\n".format(np.min(accuracies)))
                outfile.write("    Max. accuracy: {}\n".format(np.max(accuracies)))
                outfile.write("    Std. accuracy: {}\n".format(standard_deviation))
                outfile.write("    Num. exact match: {}\n".format(exact_match_distribution[True]))
                outfile.write("    Num. not exact match: {}\n\n".format(exact_match_distribution[False]))
                sheet.write(i + 1, 0, item_key)
                sheet.write(i + 1, 1, len(item_values["accuracy"]))
                sheet.write(i + 1, 2, mean_accuracy)
                sheet.write(i + 1, 3, standard_deviation)
                sheet.write(i + 1, 4, exact_match_distribution[True])
                sheet.write(i + 1, 5, exact_match_distribution[False])
                sheet.write(i + 1, 6, exact_match_distribution[True] / (
                        exact_match_distribution[False] + exact_match_distribution[True]))
            outfile.write("\n\n\n")
        outfile_excel = output_file.split(".txt")[0] + ".xls"
        workbook.save(outfile_excel)


if __name__ == "__main__":
    predictions_file = "adverb_preds/predictions_module_adverb_transform_split_test.json"
    save_file = "adverb_preds/analysis_split_test.txt"
    predictions = read_predictions(predictions_file)
    error_analysis_d, exact_matches_l, all_accuracies_l = analyze_predictions(
        predictions, module_target_key="adverb_target_tensor")
    write_error_analysis(error_analysis_d, exact_matches_l, all_accuracies_l, output_file=save_file)

