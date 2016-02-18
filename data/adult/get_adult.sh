#!/usr/bin/env sh
# This scripts downloads the adult data.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

echo "Done."
