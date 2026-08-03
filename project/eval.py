import os
from utils import (eval_parse_args, 
                   setup_reproducibility_environment, 
                   setup_reproducible_dataloader,
                   setup_experiment_folder,
                   build_run_path_components,
                   is_debug_mode,
                   Logger) 

from modeling import load_model
from benchmarking import load_benchmark
from utils.ddp import setup_ddp_environment


def main(args):

    if is_debug_mode():
        Logger.info("Running in DEBUG MODE")

    setup_reproducibility_environment(args.seed)

    if args.backend == "ddp":
        setup_ddp_environment()
    
    path_components = build_run_path_components(
        seed=args.seed,
        answer_extraction_model=args.answer_extraction_model,
        include_answer_extraction_model=args.answer_extraction_with_llm,
        budget_forcing_prompt=args.path_budget_forcing_prompt,
    )
    experiment_folder = setup_experiment_folder(output=args.output,
                                                model_name=args.model,
                                                benchmark_name=args.benchmark,
                                                experiment_name=args.experiment_name,
                                                path_components=path_components)
    
    logger = Logger(experiment_folder=experiment_folder,
                    configs=vars(args),
                    use_wandb=args.use_wandb,
                    wandb_name=args.wandb_name,
                    wandb_project=args.wandb_project)
    
    logger.info(f"Experiment folder set up at: {experiment_folder}")

    benchmark = load_benchmark(args.benchmark)

    model = load_model(model_name=args.model, 
                       backend=args.backend,
                       max_tokens=args.max_tokens,
                       max_num_images=args.max_num_images,
                       seed=args.seed,
                       additional_args=args.additional_model_args,
                       override_system_prompt=args.override_system_prompt,
                       override_reasoning_prompt=args.override_reasoning_prompt,
                       override_prepend_reasoning_prompt=args.override_prepend_reasoning_prompt)
    
    model.eval()

    if args.skip_generation and os.path.exists(f"{experiment_folder}/generations.jsonl"):
        logger.info("Output file already exists. Skipping generation step as requested!.")
    else:
        logger.info("Output file does not exist or generation not skipped. Generating outputs...")
        benchmark.generate_outputs(model=model, output_file=f"{experiment_folder}/generations.jsonl", limit=args.limit)

    logger.info("Generated outputs for the benchmark.")

    if args.answer_extraction_with_llm:
        from evaluation.answer_extractor import AnswerExtractionPipeline, process_document

        assert args.llm_endpoint is not None, "LLM endpoint must be provided when using LLM-based answer extraction."

        answer_extractor = AnswerExtractionPipeline(
            model_name=args.answer_extraction_model,
            llm_endpoint=args.llm_endpoint
        )
        process_document(input_file=f"{experiment_folder}/generations.jsonl", pipeline=answer_extractor, logger=logger)

    metrics = benchmark.evaluate(model=model, output_file=f"{experiment_folder}/generations.jsonl")

    logger.log_metrics(metrics)
    logger.show_metrics2table()
    logger.save_metrics("results.json")

    logger.terminate_experiment()


if __name__ == "__main__":
    args = eval_parse_args()

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    if args.show_models_and_benchmarks:
        from utils import show_available_models_and_benchmarks
        show_available_models_and_benchmarks(args.results_folder)
    else:
        main(args)
    
