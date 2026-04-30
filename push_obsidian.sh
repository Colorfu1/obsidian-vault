#!/bin/sh

echo "== Push Obsidian vault =="

lg2 add . >/tmp/obsidian_add.log 2>&1

status="$(lg2 status 2>/dev/null)"

case "$status" in
  *"No staged changes"*|*"nothing would be in the commit"*|*"nothing to commit"*|*"working tree clean"*)
    echo "== No changes, skip push =="
    exit 0
    ;;
esac

msg="update from iphone $(date "+%Y-%m-%d %H:%M:%S")"

lg2 commit -m "$msg" >/tmp/obsidian_commit.log 2>&1

if [ $? -eq 0 ]; then
  lg2 push origin >/tmp/obsidian_push.log 2>&1
  if [ $? -eq 0 ]; then
    echo "== Push done =="
  else
    echo "== Push failed =="
    cat /tmp/obsidian_push.log
  fi
else
  if grep -q "No staged changes" /tmp/obsidian_commit.log 2>/dev/null; then
    echo "== No changes, skip push =="
  else
    echo "== Commit failed =="
    cat /tmp/obsidian_commit.log
  fi
fi
