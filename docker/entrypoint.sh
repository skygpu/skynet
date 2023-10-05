#!/bin/sh

export VIRTUAL_ENV='/skynet/.venv'
poetry env use $VIRTUAL_ENV/bin/python

poetry install

exec poetry run "$@"
