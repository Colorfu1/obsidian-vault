#!/bin/sh
echo "== Push Obsidian vault =="
lg2 add .
msg="update from iphone $(date '+%Y-%m-%d %H:%M:%S')"
lg2 commit -m "$msg"
lg2 push origin
echo "== Push done =="
