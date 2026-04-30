#!/bin/sh

echo "== Push Obsidian vault =="

lg2 add .
lg2 status

msg="update from iphone $(date "+%Y-%m-%d %H:%M:%S")"

lg2 commit -m "$msg"

if [ $? -eq 0 ]; then
  lg2 push origin
  echo "== Push done =="
else
  echo "== No commit created, skip push =="
fi
