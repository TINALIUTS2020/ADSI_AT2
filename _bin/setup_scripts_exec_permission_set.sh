#!/bin/bash
shopt -s nullglob

containerDir="containers"

for f in $containerDir/*/setup.sh
do
    echo "f: $f"
    # check if file is executable
    if [[ -x $f ]]; then
        echo "file is executable"
    else
        echo "file is not executable"
        # make file executable
        chmod +x $f
        echo "updated permissions"
    fi
done