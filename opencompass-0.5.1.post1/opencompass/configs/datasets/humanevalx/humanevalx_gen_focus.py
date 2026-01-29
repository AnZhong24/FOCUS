from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalXDataset, HumanevalXEvaluator

humanevalx_reader_cfg = dict(
    input_columns=['prompt'], output_column='declaration', train_split='test')

humanevalx_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'),
    # prompt_template=dict(
    #     type=PromptTemplate,
    #     template=dict(round=[
    #         dict(
    #             role='HUMAN',
    #             prompt='Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function within a code block ``````:\n{prompt}'),
    #     ])),
    # prompt_template=dict(
    #     type=PromptTemplate,
    #     template='Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

# humanevalx_infer_cfg = {
#     lang: dict(
#         prompt_template=dict(
#             type=PromptTemplate,
#             template=dict(round=[
#                 dict(
#                     role='HUMAN',
#                     prompt=
#                     f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


# ### Instruction:
# Create a {lang} script for this problem:
# {{prompt}}

# ### Response:"""),
#             ])),
#         retriever=dict(type=ZeroRetriever),
#         inferencer=dict(type=GenInferencer, max_out_len=1024))
#     for lang in ['python', 'cpp', 'go', 'java', 'js']
# }

humanevalx_infer_cfg = {
    lang: dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f"""Please complete the following {lang} function according to the docstring and function signature. Your response should only contain the complete function implementation in a code block.```{lang}```\n{{prompt}}"""),
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=1024))
    for lang in ['python', 'cpp', 'go', 'java', 'js']
}

humanevalx_eval_cfg_dict = {
    lang : dict(
        evaluator=dict(
            type=HumanevalXEvaluator,
            language=lang,
            ip_address=
            'localhost',  # replace to your code_eval_server ip_address, port
            port=5000),  # refer to https://opencompass.readthedocs.io/en/latest/advanced_guides/code_eval_service.html to launch a server
        pred_role='BOT')
    for lang in ['python', 'cpp', 'go', 'java', 'js']   # do not support rust now
}

# Please download the needed `xx.jsonl.gz` from
# https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx
# and move them into `data/humanevalx/` folder
humanevalx_datasets = [
    dict(
        type=HumanevalXDataset,
        abbr=f'humanevalx-{lang}',
        language=lang,
        path='./data/humanevalx',
        reader_cfg=humanevalx_reader_cfg,
        # infer_cfg=humanevalx_infer_cfg,
        infer_cfg=humanevalx_infer_cfg[lang],
        eval_cfg=humanevalx_eval_cfg_dict[lang])
    for lang in ['python', 'cpp', 'go', 'java', 'js']
]
