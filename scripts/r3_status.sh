#!/bin/bash
# Quick status snapshot for the round-3 20M index build.
cd ~/workspace/ai-detector 2>/dev/null || exit 1
echo "=== r3-20M status $(date '+%m-%d %H:%M') ==="
for l in crossref pubmed theses; do printf "%-10s %s\n" "r3_$l:" "$(tail -1 /tmp/r3_$l.log 2>/dev/null)"; done
for w in 0 1; do printf "%-10s %s\n" "ccw$w:" "$(tail -1 /tmp/r3_ccw$w.log 2>/dev/null)"; done
printf "%-10s %s\n" "wiki:" "$(tail -1 /tmp/r3_wiki.log 2>/dev/null)"
for t in 0 1; do printf "%-10s %s\n" "pm2_$t:" "$(tail -1 /tmp/r3_pm2$t.log 2>/dev/null)"; done
printf "%-10s %s\n" "merge:" "$(tail -1 /tmp/r3_merge.log 2>/dev/null)"
echo "disk free: $(df -BG data 2>/dev/null | tail -1 | awk '{print $4}')"
echo "build DB : $(du -sh data/mega_index/plag_index.db.build 2>/dev/null | cut -f1)"
echo "shards   : $(ls data/oa_corpus/fp_ccw_*.jsonl data/oa_corpus/fp_pubmed2_*.jsonl data/oa_corpus/fp_wiki_*.jsonl 2>/dev/null | wc -l) pending"
echo "workers  : $(pgrep -c -f 'scripts/r3_' 2>/dev/null) python leg processes"
