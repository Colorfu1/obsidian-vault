#!/bin/sh

echo "== Push Obsidian vault =="

lg2 add . >/dev/null 2>&1

status="$(lg2 status 2>/dev/null)"

case "$status" in
  *"No staged changes"*|*"nothing would be in the commit"*|*"nothing to commit"*|*"working tree clean"*)
    echo "== No changes, skip push =="
    exit 0
    ;;
esac

msg="update from iphone $(date "+%Y-%m-%d %H:%M:%S")"

lg2 commit -m "$msg" >/dev/null 2>&1

if [ $? -eq 0 ]; then
  lg2 push origin >/dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "== Push done =="
  else
    echo "== Push failed =="
  fi
else
  echo "== No changes or commit failed, skip push =="
fi
