# Single GPU 
# Full 8b
export ZE_AFFINITY_MASK=4 # enable when single_device

Run_llama3-8b_single_device_full() {
    tune run full_finetune_single_device \
    --config llama3/8B_full_single_device \
    batch_size=4 \
    gradient_accumulation_steps=1 \
    dataset.max_seq_len=2048 \
    dataset.packed=True \
    log_peak_memory_stats=True \
    enable_activation_checkpointing=True \
    device=xpu \
    optimizer._component_=torch.optim.AdamW \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original \
    tokenizer.path=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/shiyan/torchtune/recipes/alpaca-llama3-finetune\
    2>&1 | tee alpaca-llama3-finetune/llama3-8b_single_device_full_bs_4_seqlen_2048.log
    
}


# Full 8b with low PagedAdamw

Run_llama3-8b_single_device_full_pagedadamw() {
    tune run full_finetune_single_device \
    --config llama3/8B_full_single_device \
    device=xpu \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original \
    tokenizer.path=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/shiyan/torchtune/recipes/alpaca-llama3-finetune \
    2>&1 | tee alpaca-llama3-finetune/llama3-8b_single_device_full_pagedadamw_bs_4_seqlen_2048.log
}



# LoRA 8b
Run_llama3-8b_single_device_lora() {
    tune run lora_finetune_single_device \
    --config llama3/8B_lora_single_device \
    batch_size=4 \
    gradient_accumulation_steps=1 \
    dataset.max_seq_len=2048 \
    dataset.packed=True \
    log_peak_memory_stats=True \
    enable_activation_checkpointing=True \
    device=xpu \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original \
    tokenizer.path=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/shiyan/torchtune/recipes/alpaca-llama3-finetune \
    2>&1 | tee alpaca-llama3-finetune/llama3-8b_single_device_lora_bs_4_seqlen_2048_test_delogits.log
}


# # QLora 8b
Run_llama3-8b_single_device_qlora() {
    tune run lora_finetune_single_device \
    --config llama3/8B_qlora_single_device \
    checkpointer.checkpoint_dir=/data2/zhuhong/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/original \
    tokenizer.path=/data2/zhuhong/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    checkpointer.output_dir=/data2/zhuhong/huggingface \
    output_dir=/home/pt-gpu/zhuhong/torchtune/recipes/alpaca-llama3-finetune
}

# distributed full
Run_llama3-8b_distributed_full() {
    tune run --nproc_per_node 4 full_finetune_distributed \
    --config llama3/8B_full \
    batch_size=4 \
    dataset.max_seq_len=2048 \
    dataset.packed=True \
    gradient_accumulation_steps=1 \
    log_peak_memory_stats=True \
    enable_activation_checkpointing=True \
    device=xpu \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original \
    tokenizer.path=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/shiyan/torchtune/recipes/alpaca-llama3-finetune \
    2>&1 | tee alpaca-llama3-finetune/llama3-8b_distributed_full_bs_4_seqlen_2048_node_4.log
}


# distributed lora
Run_llama3-8b_distributed_lora() {
    tune run --nproc_per_node 4 lora_finetune_distributed \
    --config llama3/8B_lora \
    batch_size=4 \
    dataset.max_seq_len=2048 \
    dataset.packed=True \
    gradient_accumulation_steps=1 \
    log_peak_memory_stats=True \
    enable_activation_checkpointing=True \
    device=xpu \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original \
    tokenizer.path=/workspace1/huggingface/hub/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/shiyan/torchtune/recipes/alpaca-llama3-finetune \
    2>&1 | tee alpaca-llama3-finetune/llama3-8b_distributed_lora_bs_4_seqlen_2048_node_4.log
}


main() {
#   Run_llama3-8b_single_device_full
#   Run_llama3-8b_single_device_full_pagedadamw
  Run_llama3-8b_single_device_lora
  # Run_llama3-8b_single_device_qlora
#   Run_llama3-8b_distributed_full
#   Run_llama3-8b_distributed_lora
}

main
