for i, (split_result, reserved_stopword_tokens) in enumerate(zip(split_results, reserved_stopwords)):
    head_text, tail_text, head_tokens, tail_tokens = split_result
    print(f"Iteration #{i}: head_text: {head_text}, tail_text: {tail_text}, head_tokens: {head_tokens}, tail_tokens: {tail_tokens}")
    cleaned_head_tokens = [t.replace(cw_aug_ins_xlmr.model.get_subword_prefix(), '') for t in head_tokens]
    print(cleaned_head_tokens)
