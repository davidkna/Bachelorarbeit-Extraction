#!/usr/bin/env pwsh

Set-StrictMode -Version Latest

$source_data = "$PWD/ner-data"
$train = "$PWD/train.spacy"
$test = "$PWD/test.spacy"

if (!(Test-Path "$source_data")) {
    New-Item $source_data -ItemType Directory
    Expand-Archive -LiteralPath '../vendor/Legal-Entity-Recognition/data/dataset_courts.zip' -DestinationPath "$source_data"
}

if (!(Test-Path $train)) {
    spacy convert --converter ner -l de "$source_data/bag.conll" $PWD
}

if (!(Test-Path $test)) {
    spacy convert --converter ner -l de "$source_data/bgh.conll" $PWD
}
