#!/bin/sh

MINLINTSCORE=10

if ! (pylint --fail-under=$MINLINTSCORE --ignore-paths=venv_* main.py src); then
    echo "PYLINT ERROR: score below required lint score"
    exit 123
else
    echo "PYLINT SUCCESS!!"
fi

echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"