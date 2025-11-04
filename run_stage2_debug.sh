#!/usr/bin/env bash

# Optional token-merge environment toggles (used only when ENABLE_TOKEN_MERGE=true)
ENABLE_TOKEN_MERGE=${ENABLE_TOKEN_MERGE:-false}
TOKEN_MERGE_ROUTES=${TOKEN_MERGE_ROUTES:-}
TOKEN_MERGE_DEFAULT_RATIO=${TOKEN_MERGE_DEFAULT_RATIO:-}
TOKEN_MERGE_SEED=${TOKEN_MERGE_SEED:-42}
TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION=${TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION:-2}

if [[ "${ENABLE_TOKEN_MERGE}" == "true" ]]; then
    export TOKEN_MERGE_ROUTES
    export TOKEN_MERGE_DEFAULT_RATIO
    export TOKEN_MERGE_SEED
    export TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION
    bash finetune/train_ddp_one_s2_token_merge.sh
else
    bash finetune/train_ddp_one_s2_debug.sh
fi
