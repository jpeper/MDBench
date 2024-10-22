from dotenv import load_dotenv
from pipeline import Pipeline
from utils import get_args, dump_output, ENV_PATH


def main(args):
    load_dotenv(ENV_PATH)

    pipeline = Pipeline(args)
    outputs, prompts = pipeline.get_model_output()
    
    infer_outputs = {}
    for output, prompt in zip(outputs, prompts):
        infer_output = {"model_response": output}
        
        example_id = prompt["example_id"]
        del prompt["example_id"]

        infer_output.update(prompt)
        infer_outputs[example_id] = infer_output

    dump_output(args.output_file, infer_outputs, append=(args.num_to_append > 0))


if __name__ == "__main__":
    args = get_args()
    main(args)
