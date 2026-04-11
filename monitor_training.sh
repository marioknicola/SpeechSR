#!/bin/bash
# Run this in a terminal to watch training progress.
# Usage:  bash monitor_training.sh

OUTPUT_DIR="/Users/marioknicola/MSc Project/SpeechSR/outputs/gan_v2"

while true; do
    clear
    echo "=== SpeechSR V2 Training Monitor — $(date) ==="
    echo ""
    if pgrep -f train_gan.py > /dev/null; then
        echo "Status : RUNNING"
        CPU=$(ps aux | grep train_gan | grep -v grep | awk 'NR==1{print $3}')
        echo "CPU    : ${CPU}%"
        ELAPSED=$(ps aux | grep train_gan | grep -v grep | awk 'NR==1{print $10}')
        echo "Elapsed: ${ELAPSED}"
    else
        echo "Status : STOPPED"
    fi
    echo ""
    echo "Output files:"
    ls -lh "$OUTPUT_DIR" 2>/dev/null
    echo ""
    echo "Training log (last 25 lines):"
    tail -25 "$OUTPUT_DIR/training.log" 2>/dev/null || echo "  (log buffering — will appear once tee flushes)"
    echo ""
    echo "Pretrain history (last saved):"
    python3 -c "
import json, sys
try:
    with open('$OUTPUT_DIR/pretrain_history.json') as f:
        h = json.load(f)
    last = h[-1]
    print(f'  Epochs done: {last[\"epoch\"]}  train={last[\"train_loss\"]:.4f}  val={last[\"val_loss\"]:.4f}')
except:
    print('  Not yet written.')
" 2>/dev/null
    echo ""
    echo "GAN history (last saved):"
    python3 -c "
import json, sys
try:
    with open('$OUTPUT_DIR/gan_history.json') as f:
        h = json.load(f)
    last = h[-1]
    print(f'  GAN epochs done: {last[\"epoch\"]}  G={last[\"g_total\"]:.4f}  D={last[\"d_loss\"]:.4f}  val={last[\"val_loss\"]:.4f}')
except:
    print('  Not yet started.')
" 2>/dev/null
    echo ""
    echo "Refreshing every 30s — Ctrl+C to exit."
    sleep 30
done
