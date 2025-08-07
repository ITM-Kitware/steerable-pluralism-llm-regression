import outlines
import json

@outlines.prompt
def unaligned_system_prompt():
    """
    You are an assistant specialized in answering multiple-choice \
    questions. When a question is posed, select the best response \
    and provide one sentence of reasoning.
    """

@outlines.prompt
def baseline_system_prompt(target):
    """
    You are an assistant specialized in answering multiple-choice \
    questions. When a question is posed, select response that best \
    meets the following criteria and provide one sentence of reasoning.\
    
    Select the response that:
    {% for attribute, value in target.items() %}
        {% if value == 1.0 -%}
            - Strongly demonstrates {{ attribute }}
        {% endif %}
        {% if value < 1.0 and value > 0.67 -%}
            - Demonstrates {{ attribute }}
        {% endif %}
        {% if value == 0.67 -%}
            - Somewhat demonstrates {{ attribute }}
        {% endif %}
        {% if value == 0.5 -%}
            - Is neutral with respect to {{ attribute }}
        {% endif %}
        {% if value == 0.33 -%}
            - Somewhat demonstrates a lack of {{ attribute }}
        {% endif %}
        {% if value < 0.3 and value > 0.0 -%}
            - Demonstrates a lack of {{ attribute }}
        {% endif %}
        {% if value == 0.0 -%}
            - Strongly demonstrates a lack of {{ attribute }}
        {% endif %}
    {% endfor %}
    """

@outlines.prompt
def baseline_prompt(question, multiple_choices):
    """
    Question:
    {{ question }}

    Responses:
    {% for choice_letter, choice in multiple_choices.items() %}
    - {{choice_letter}}: {{ choice }}
    {% endfor %}

    Provide the letter of your selected response with one sentence of reasoning.
    """

def baseline_json_schema(letters):
    json_schema = {
        "properties": {
        "reasoning": {
            "title": "Reasoning",
            "type": "string",
            "minLength": 1,
            "maxLength": 512
            },
        "choice": {
            "title": "Choice",
            "enum": letters
            }
        },
     "required": ["reasoning","choice"],
     "type": "object"
    }
    return json.dumps(json_schema)


@outlines.prompt
def regression_system_prompt(attribute, attribute_description, attribute_scale_factor):
    """
    You are an assistant tasked with judging responses to questions. \
    Given a question and response, your job is to assess (on a scale from {{ "%g" | format(0 * attribute_scale_factor) }} \
    to {{ "%g" | format(1.0 * attribute_scale_factor) }}) how well the response demonstrates {attribute}}.
    
    {{attribute_description}}

    A score of {{ "%g" | format(0 * attribute_scale_factor) }} means that the response demontsrates \
    a strong lack of {{attribute}}. \
    A score of {{ "%g" | format(0.5 * attribute_scale_factor) }} means that the response is neutral \
    with respect to {{attribute}}. \
    A score of {{ "%g" | format(1.0 * attribute_scale_factor) }} means that the response perfectly \
    demonstrates {{attribute}}. \

    Provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def regression_prompt(attribute, question, response):
    """
    Question:
    {{ question }}

    Response:
    {{ response }}

    Provide a score of how well the response demonstrates {{attribute}} with one sentence of reasoning.
    """

def regression_json_schema(attribute_scale_factor):
    json_schema = {
        "properties": {
            "reasoning": {
                "title": "Reasoning",
                "type": "string",
                "minLength": 1,
                "maxLength": 512
                },
                "score": {
                    "type": "integer",
                    "minimum": 0 * attribute_scale_factor,
                    "maximum": 1 * attribute_scale_factor
                }
            },
        "required": ["reasoning","score"],
        "title": "ScoreRegression",
        "type": "object"
    }
    return json.dumps(json_schema)

@outlines.prompt
def comparative_regression_system_prompt(attribute, attribute_description, attribute_scale_factor):
    """
    You are an assistant tasked with judging responses to questions. \
    Given a question and a list of multiple choice responses, your job is to assess (on a scale from \
    {{ "%g" | format(0 * attribute_scale_factor) }} to {{ "%g" | format(1.0 * attribute_scale_factor) }}) \
    how well each response demonstrates {attribute}}.
    
    {{attribute_description}}

    A score of {{ "%g" | format(0 * attribute_scale_factor) }} means that the response demontsrates \
    a strong lack of {{attribute}}. \
    A score of {{ "%g" | format(0.5 * attribute_scale_factor) }} means that the response is neutral \
    with respect to {{attribute}}. \
    A score of {{ "%g" | format(1.0 * attribute_scale_factor) }} means that the response perfectly \
    demonstrates {{attribute}}. \

    For each response, provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def comparative_regression_prompt(attribute, question, multiple_choices):
    """
    Question:
    {{ question }}

    Responses:
    {% for choice_letter, choice in multiple_choices.items() %}
    - {{choice_letter}}: {{ choice }}
    {% endfor %}

    For each choice, provide a score of how well the response demonstrates {{attribute}} \
    with one sentence of reasoning.
    """

def comparative_regression_json_schema(choices, attribute_scale_factor):
    json_schema = {
        "type": "object",
        "properties": {
            choice: {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 0 * attribute_scale_factor,
                        "maximum": 1 * attribute_scale_factor
                    }
                },
                "required": ["score", "reasoning"]
            }
            for choice in choices
        },
        "required": list(choices)
    }
    return json.dumps(json_schema)