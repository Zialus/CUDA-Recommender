#!/bin/bash
set -x

rawA="coisas.rmse"

grep "Wall-time" $1 | cut -d':' -f2 | cut -d' ' -f2 >> ${rawA}
grep "Test RMSE" $1 | cut -d'=' -f2 | cut -d' ' -f2 >> ${rawA}
