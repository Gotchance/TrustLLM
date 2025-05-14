from trustllm.generation.generation import LLMGeneration

llm_gen = LLMGeneration(
    # model_path="meta-llama/Llama-2-7b-chat-hf",
    model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    test_type="safety",
    data_path="dataset"
)

llm_gen.generation_results()
