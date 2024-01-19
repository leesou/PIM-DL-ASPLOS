#!/bin/sh

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

echo "-------------------- sensitivity analysis start --------------------"

cd sensitivity-analysis
python run_experiments.py
cd ../

mkdir -p results/sensitivity-analysis/
cp -r sensitivity-analysis/plot/* results/sensitivity-analysis/

echo "-------------------- sensitivity analysis finish --------------------"
