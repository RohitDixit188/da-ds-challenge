# A data set for a particular model has a large number of features. We suspect that not all of them are useful.
# Gathering the feature scores for the resulting model has confirmed our suspicions. Feature extraction for
# this data set takes a long time but training the model is fairly quick. We need to deliver the model soon, so we've
# decided to just remove from the data set the features that have no value, and then retrain the model.
#
# This script removes all feature data which has been found to add no value. It takes a file containing the original
# feature data and, using a file containing scores for each feature with a score > 0, creates a new feature file with
# the masked feature data.
#
# Assumptions about the feature data:
# 1. Feature data row format is a comma-separated list:
#           ID, label, feature 0 value,..., feature n value
#    where all feature values are float
# 2. All rows have the same number of values
# 3. The resulting output file must contain the ID and label in addition to the masked features
#
# Assumptions about the feature score data:
# 1. A feature score row is of the following format:
#       f<0-based feature number> <score>
#   For example:
#       f590 1912.0991193927057
# 2. Only features with a non-zero score will be present in the file
# 3. The rows in the file are ordered by highest-scoring feature to lowest non-zero-scoring feature

import argparse
import pandas as pd
from pandas.core.frame import DataFrame #since it was not specified, I installed pandas v1.3.1

def apply_feature_mask(feature_data_file, feature_score_file, output_file):
    
    # loading feature_data_file into dataframe object
    feature_data_file : DataFrame = pd.read_csv(feature_data_file, encoding = "utf-16")

    # importing feature_score_file into dataframe object
    feature_score_file : DataFrame = pd.read_csv(feature_score_file, names=["feature_number","score"], delimiter=r"\s+")

    # remove the 'f's from the feature score id
    feature_score_file['feature_number'] = feature_score_file['feature_number'].str.replace('f', '')
    # convert feature id to number so we can sort later
    feature_score_file['feature_number'] = pd.to_numeric(feature_score_file['feature_number'], errors='coerce')
    # convert feature score to number so we can filter out features with score 0
    feature_score_file['score'] = pd.to_numeric(feature_score_file['score'], errors='coerce')
    # sort feature ID in ascending order. This will allow us to use them as array to keep the columns
    # in the feature data file
    feature_score_file.sort_values(by=['feature_number'], inplace=True, ignore_index=True)

    # remove features with score 0
    feature_score_file = feature_score_file[feature_score_file['score'] > 0]

    # get the columns with feature ids
    columns_to_keep = feature_score_file['feature_number']

    # since the feature IDs in the feature data file starts from the columns with index 2
    # with add an offset here of 2 to the feature IDs list
    #   ID, label, f0, f1, f2
    columns_to_keep = 2 + columns_to_keep

    # convert to a regular python list
    columns_to_keep = columns_to_keep.tolist()

    # now we add the first two indexes so we keep also the ID and label
    columns_to_keep = [0, 1] + columns_to_keep

    # select only the columns we're interested
    masked_feature_data_file = feature_data_file[feature_data_file.columns[columns_to_keep]]

    # save to file
    masked_feature_data_file.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--feature-score-file', type=str, required=True, help='File containing feature scores'
    )
    parser.add_argument(
        '-f', '--feature-data-file', type=str, required=True, help='File containing unmasked feature data'
    )
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='Output file containing masked feature data')
    args = parser.parse_args()

    kwargs = {}
    kwargs['feature_data_file'] = args.feature_data_file
    kwargs['feature_score_file'] = args.feature_score_file
    kwargs['output_file'] = args.output_file

    apply_feature_mask(**kwargs)
    
    
#python feature_mask.py -s feature_importance.txt -f test.csv -o output.csv #Validation
