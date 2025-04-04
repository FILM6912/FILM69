from unsloth import UnslothVisionDataCollator

def vision_data_collator_format(
    model,
    processor,
    train_on_responses_only=False,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
):

    return UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only=train_on_responses_only,
        instruction_part=instruction_part,
        response_part=response_part,
    )

def vision_data_collator(
    model,
    processor,

):

    return UnslothVisionDataCollator(
        model,
        processor,
    )

