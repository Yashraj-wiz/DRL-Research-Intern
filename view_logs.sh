#!/bin/bash
# Automatically tails the docker logs and filters out the TF_REPEATED_DATA spam
echo "Tailing logs (Spam filtered). Press Ctrl+C to stop."
docker logs -f gotgtrl_training 2>&1 | grep --line-buffered -v "TF_REPEATED_DATA"
