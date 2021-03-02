#!/usr/bin/env pwsh

Set-StrictMode -Version Latest

spacy init fill-config base_config.cfg config.cfg
spacy train config.cfg --paths.train ./bag.spacy --paths.dev ./bag.spacy