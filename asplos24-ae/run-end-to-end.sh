#!/bin/bash

function lockfile(){
	lock_file=$1

	if [ -f "${lock_file}" ]; then
		echo "Lock file already exists. Exiting..."
		exit 1
	fi

	trap "echo finish execution; rm -rf ${lock_file}; rm -rf '../${lock_file}'; exit" EXIT

	touch "${lock_file}"
}

mkdir -p results

lockfile "results/lock_file"

echo "-------------------- end-to-end start --------------------"

cd ./end-to-end/
python run_experiments.py
cd ../

mkdir -p results/end-to-end/
cp -r end-to-end/plot/* results/end-to-end/

echo "-------------------- end-to-end finish --------------------"
