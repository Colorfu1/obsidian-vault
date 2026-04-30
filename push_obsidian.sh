#!/bin/sh

echo "== Push Obsidian vault =="

lg2 add .

status="$(lg2 status)"
echo "$status"

case "$status" in
  *"No changes"*|*"nothing to commit"*|*"working tree clean"*|*"No staged changes"*)
    echo "== No changes, skip commit and push =="
    exit 0
    ;;
esac

msg="update from iphone $(date "+%Y-%m-%d %H:%M:%S")"

lg2 commit -m "$msg"

if [ $? -ne 0 ]; then
  echo "== Commit failed, skip push =="
  exit 1
fi

lg2 push origin

echo "== Push done =="
