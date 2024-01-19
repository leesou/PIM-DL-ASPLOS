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

echo "-------------------- search space illustration start --------------------"

cd search-space-illustration
python run_experiments.py
cd ../

mkdir -p results/search-space-illustration/
cp -r search-space-illustration/plot/* results/search-space-illustration/

echo "-------------------- search space illustration finish --------------------"
