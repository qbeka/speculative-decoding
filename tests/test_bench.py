from specdec.bench import request_seed


def test_every_prompt_and_repeat_gets_its_own_seed():
    prompts = [f"test-hotel-{index:04d}" for index in range(30)]
    seeds = {request_seed(prompt, repeat) for prompt in prompts for repeat in range(3)}
    assert len(seeds) == 90


def test_a_seed_does_not_depend_on_the_arm():
    assert request_seed("test-hotel-0001", 2) == request_seed("test-hotel-0001", 2)
