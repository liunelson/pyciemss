#!/bin/bash

INCLUDED_NOTEBOOKS="docs/source/*.ipynb"

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS