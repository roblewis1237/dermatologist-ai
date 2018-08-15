#!/bin/bash
# This script downloads and unzips the skin lesion datasets
# Run with filename in linux, and with sh {filename} on windows
# TODO: could put some defensive code in here (e.g. delete data/ if exists before mkdir data)

# Make a directory for the data
mkdir data
cd data

# Download datasets
curl -O https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip
curl -O https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip
curl -O https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip

# Unzip the datasets
unzip train.zip
unzip valid.zip
unzip test.zip

# Delete the zip files
rm -rf train.zip
rm -rf valid.zip
rm -rf test.zip
