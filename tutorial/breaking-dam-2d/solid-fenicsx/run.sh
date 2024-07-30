#!/bin/bash
set -e -u

exec > >(tee --append "$LOGFILE") 2>&1

python3 solid.py

close_log
