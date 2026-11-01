R=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
S=$R/outputs_final
sec() { echo; echo "=== $1 ==="; }
chk() { n=$(ls $S/shards/$1_$2_$3_*.jsonl.gz 2>/dev/null | wc -l); echo "$1 $2 $3: $n of $4"; }
sec my_jobs
squeue -u manik -o "%.9i %.34j %.8T %.22P %.8M %R" | grep -E " j[012]"
sec finished_jobs
sacct -j $(cut -d' ' -f1 $S/jobs.log | paste -sd,) -X --format=JobID,JobName%34,Partition%14,State,Elapsed,ExitCode
sec shards_done
for m in maira2 radvlm medgemma chex chexagent; do chk j0a $m uniform 1; chk j0b $m uniform 1; done
chk j2 maira2 flip 5; chk j2 radvlm flip 12; chk j2 medgemma flip 15; chk j2 chex flip 1; chk j2 chexagent flip 10
chk j1 maira2 native 5; chk j1 radvlm native 12; chk j1 medgemma native 15; chk j1 chexagent native 10
chk j3 medgemma uniform 15
sec running_parts
ls -la $S/shards/*.part 2>/dev/null | awk '{print $5, $9}' | sed 's#/home.*/##'
sec shard_meta
for f in $S/shards/*.jsonl.gz; do [ -f "$f" ] && echo "$(basename $f) $(zcat $f | tail -n 1 | sed -E 's/.*"n": ([0-9]+), "errors": ([0-9]+), "seconds": ([0-9.]+),.*"gpu": "([^"]*)".*/n=\1 err=\2 sec=\3 \4/')"; done
sec log_tails
for f in $(ls -t $S/logs/j*.out 2>/dev/null | head -6); do echo "-- $(basename $f)"; tail -n 2 $f | cut -c1-200; done
sec tracebacks
grep -l Traceback $S/logs/j*.out 2>/dev/null
sec quota
lfs quota -h /home/manik | sed -n '3p'
echo; echo STATUS DONE
