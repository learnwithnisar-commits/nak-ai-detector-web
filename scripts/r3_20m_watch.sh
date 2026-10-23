#!/bin/bash
# Watch for 10M build completion, then restart fleet for 20M target.
while true; do
  if [ -f /tmp/r3_FINAL_DONE ]; then
    # 10M build done; clear cap flag so supervisor restarts for 20M
    rm -f /tmp/r3_CAP_REACHED /tmp/r3_FINAL_DONE
    echo "$(date): 10M build done, cap cleared for 20M restart" >> /tmp/r3_20m_restart.log
    break
  fi
  # also check if build died without FINAL_DONE
  if ! pgrep -f "build_mega_db.py" > /dev/null; then
    # build not running and no FINAL_DONE -> it may have failed; clear cap anyway
    # (supervisor will restart legs; finalize cron will retry build at 20M)
    rm -f /tmp/r3_CAP_REACHED
    echo "$(date): build not running, cap cleared for 20M" >> /tmp/r3_20m_restart.log
    break
  fi
  sleep 300
done
