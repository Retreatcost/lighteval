# MIT License

# Copyright (c) 2026 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from lighteval.logging.info_loggers import MetricsLogger
from lighteval.metrics import apply_metric
from lighteval.metrics.metrics_sample import JudgeLLMMTBench
from lighteval.metrics.utils.llm_as_judge import JudgeLM
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.models.vllm.vllm_model import VLLMModel
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod


def _template(question, answer, options, gold):
    return [{"role": "user", "content": question}]


def _process_response(response):
    return 4


def _prompt(line, task_name):
    return Doc(query=line["query"], choices=[], gold_index=[], task_name=task_name)


def _doc(index):
    return Doc(
        task_name="mt_bench",
        query=f"question {index * 2 + 1}",
        choices=[],
        gold_index=[],
        specific={
            "multi_turn_queries": [f"question {index * 2 + 1}", f"question {index * 2 + 2}"],
            "reference": [],
        },
        sampling_methods=[SamplingMethod.GENERATIVE],
        generation_size=10,
        stop_sequences=[],
    )


def _metric_group():
    metric = object.__new__(JudgeLLMMTBench)
    metric.judge = SimpleNamespace(
        model="judge",
        backend="vllm",
        template=_template,
        process_judge_response=_process_response,
        evaluate_answer_batch=Mock(return_value=([1, 2, 3, 4], ["prompt"] * 4, ["judgment"] * 4)),
    )
    return SampleLevelMetricGrouping(
        metric_name=["judge_score_turn_1", "judge_score_turn_2"],
        higher_is_better={"judge_score_turn_1": True, "judge_score_turn_2": True},
        category=SamplingMethod.GENERATIVE,
        batched_compute=True,
        sample_level_fn=metric,
        corpus_level_fn={"judge_score_turn_1": np.mean, "judge_score_turn_2": np.mean},
    )


def _task_config(metric_group):
    return LightevalTaskConfig(
        name="mt_bench_test",
        prompt_function=_prompt,
        hf_repo="unused",
        hf_subset="default",
        evaluation_splits=["train"],
        metrics=[metric_group],
    )


def test_mt_bench_batches_both_turns_per_document():
    metric_group = _metric_group()
    responses = [ModelResponse(text=[f"answer {index * 2 + 1}", f"answer {index * 2 + 2}"]) for index in range(2)]

    outputs = apply_metric(responses=responses, docs=[_doc(0), _doc(1)], metrics=[metric_group])

    judge_call = metric_group.sample_level_fn.judge.evaluate_answer_batch.call_args
    assert len(judge_call.kwargs["questions"]) == 4
    assert json.loads(judge_call.kwargs["questions"][1]) == {"query": "question 2", "context": "answer 1"}
    assert [output["judge_score_turn_1"] for output in outputs] == [1, 3]
    assert [output["judge_score_turn_2"] for output in outputs] == [2, 4]


def test_mt_bench_diagnostic_fields_skip_aggregation():
    task = LightevalTask(_task_config(_metric_group()))
    logger = MetricsLogger()
    logger.log(
        task.full_name,
        {
            "judge_score_turn_1": 4,
            "judge_score_turn_2": 5,
            "user_prompt": ["prompt 1", "prompt 2"],
            "judgement": ["judgment 1", "judgment 2"],
        },
    )

    logger.aggregate({task.full_name: task}, bootstrap_iters=0)

    assert logger.metric_aggregated[task.full_name]["judge_score_turn_1"] == 4
    assert logger.metric_aggregated[task.full_name]["judge_score_turn_2"] == 5
    assert logger.metrics_values[task.full_name]["user_prompt"] == [["prompt 1", "prompt 2"]]


def test_mt_bench_task_serialization_excludes_object_addresses():
    serialized = str(_task_config(_metric_group()))

    assert "0x" not in serialized
    assert "JudgeLLMMTBench(model=judge, backend=vllm" in serialized
    assert "'judge_score_turn_1': 'mean'" in serialized


def test_vllm_judge_uses_plain_token_ids():
    judge = JudgeLM(
        model="judge",
        templates=_template,
        process_judge_response=_process_response,
        judge_backend="vllm",
    )
    pipe = Mock()
    pipe.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="judgment")])]
    judge.pipe = pipe
    judge.tokenizer = Mock()
    judge.tokenizer.apply_chat_template.return_value = [1, 2, 3]
    judge.sampling_params = Mock()

    scores, _, _ = judge.evaluate_answer_batch(
        questions=["question"], answers=["answer"], options=[None], golds=[None]
    )

    assert scores == [4]
    judge.tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "question"}], return_dict=False, add_generation_prompt=True
    )
    assert pipe.generate.call_args.kwargs["prompts"] == [{"prompt_token_ids": [1, 2, 3]}]


def test_vllm_generates_second_turn_with_first_answer_as_history():
    model = object.__new__(VLLMModel)
    model.use_chat_template = True
    model._max_length = 4096
    model._add_special_tokens = False
    model.config = SimpleNamespace(generation_parameters=SimpleNamespace(max_new_tokens=10))
    model.DATASET_SPLITS = 1
    model._tokenizer = Mock()
    model._tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: json.dumps(messages)
    model._tokenizer.side_effect = lambda contexts, **kwargs: {"input_ids": [[1] for _ in contexts]}
    model.prompt_manager = PromptManager(True, model._tokenizer)
    model._generate = Mock(
        side_effect=[
            [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(text="answer 1", token_ids=[1])])],
            [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(text="answer 2", token_ids=[2])])],
        ]
    )

    responses = model._greedy_until([_doc(0)])

    assert responses[0].text == ["answer 1", "answer 2"]
    assert responses[0].output_tokens == [[1], [2]]
    assert model._tokenizer.apply_chat_template.call_args_list[1].args[0] == [
        {"role": "user", "content": "question 1"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "question 2"},
    ]
