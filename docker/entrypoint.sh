#!/bin/sh

uv sync

exec uv run "$@"
