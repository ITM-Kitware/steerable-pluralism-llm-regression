def get_example_reasoning(attribute, value):
    # For Helpsteer2 use Likert-5 scale statements
    if attribute in ["correctness","coherence","complexity","verbosity","helpfulness"]:
        if value == 0.0:
            reasoning = f'The response is strongly not characterized by {attribute}.'
        if value == 0.25:
            reasoning = f'The response is not characterized by {attribute}.'
        elif value == 0.5:
            reasoning = f'The response is neither characterized nor not characterized by {attribute}.'
        elif value == 0.75:
            reasoning = f'The response is characterized by {attribute}.'
        else:
            reasoning = f'The response is strongly characterized by {attribute}.'
    elif attribute in ['care','fairness','liberty','loyalty','authority','sanctity']:
        if value == 1.0:
            reasoning = f'The response strongly demonstrates {attribute}.'
        elif value == 0.83:
            reasoning = f'The response demonstrates {attribute}.'
        elif value == 0.67:
            reasoning = f'The response somewhat demonstrates {attribute}.'
        elif value == 0.5:
            reasoning = f'The response is neutral with respect to {attribute}.'
        elif value == 0.33:
            reasoning = f'The response somewhat demonstrates a lack of {attribute}.'
        elif value == 0.17:
            reasoning = f'The response demonstrates a lack of {attribute}.'
        else:
            reasoning = f'The response strongly demonstrates a lack of {attribute}.'
    else:
        raise RuntimeError(f"No example reasoning for {attribute}")
    return reasoning
