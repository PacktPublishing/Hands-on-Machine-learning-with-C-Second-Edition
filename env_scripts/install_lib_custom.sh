#!/usr/bin/env bash
set -x
set -e

START_DIR=$(pwd)
REPOSITORY=$1
COMMIT_HASH=$2
shift; shift;
EXTRA_COMMAND=$@

cd $START_DIR/libs/sources
git clone $REPOSITORY
cd "$(basename "$REPOSITORY" .git)"
git checkout $COMMIT_HASH

if [ -f ".gitmodules" ];then
sed -i 's/git:\/\//https:\/\//g' ".gitmodules"
fi

git submodule update --init --recursive
$EXTRA_COMMAND
cd ..
rm -rf build
cd $START_DIR

