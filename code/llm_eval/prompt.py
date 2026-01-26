"""Module to get system and user prompts for LLM evaluations."""

from enum import Enum

FACT_CHECK_SYSTEM_PROMPT = """You are a professional fact checker, your task is to classify whether the given claim is true or false based on the evidence text provided.
"""  # noqa: E501

FEW_SHOT_EXAMPLES = """"Here are some example:
Claim: 150,000 (oil spill) claims have been filed by Floridians, but only 40 percent of them have been paid.
Evidence: In her last meeting of the Florida Cabinet, state Chief Financial Officer and unsuccessful gubernatorial candidate Alex Sink took BP claims administrator Ken Feinberg to task for failing to live up to his promise to swiftly respond to oil spill claims from Floridians and Florida business owners. Sink, who has been critical of BP and the handling of the claims process, said Floridians are having their claims denied without being given a reason, that people in similar situations are having their claims processed differently, and that Feinberg has been \"incredibly unresponsive to Floridians and to Florida business owners. \"It just reveals that this process continues to be inefficient,\" Sink said near the end of a five-hour Cabinet meeting on Dec. 7, 2010. \"One hundred, fifty thousand claims have been filed by Floridians, but only 40 percent of them have been paid, a third of them have been denied. and then we have 20 percent that are still in limbo land. \"Now, more than seven months after the initial Deepwater Horizon oil rig explosion, we wanted to see if Sink is describing the speed of the claims process correctly. To recap, President Barack Obama selected Feinberg to manage the disbursement of $20 billion provided by BP to compensate spill victims through something called the Gulf Coast Claims Facility. Feinberg has been criticized by politicians at the state and national level for months for what some politicians have called a much too slow response. Sink spokesman Kevin Cate pointed us to the Gulf Coast Claims Facility's own analysis as evidence of the slow claims process. Cate used the latest statistics available for Dec. 7. We should note that as of Dec. 7, Feinberg has awarded more than $900 million to Floridians and Florida business owners. Almost all of the money paid out is compensation for lost earnings or profits. You can read the latest report for yourself here. In lashing out against BP claims administrator Feinberg, Sink noted that of all claims made in Florida following the Gulf of Mexico oil spill, only 40 percent had been paid. She's just about right on. So far, about 43 percent of claims submitted by Floridians or Florida business owners have been paid, another 39 percent have been denied and a little more than 18 percent are still under some type of review.
{"label": true}

Claim: Leonardo DiCaprio has donated $10,000,000 to the International Visegrad Fund to support Ukraine.
Evidence: The International Visegrad Fund denies Leonardo DiCaprio has donated in support of Ukraine, but he has donated to other humanitarian organizations. Leonardo DiCaprio is a well-known American actor, film producer, and environmentalist who has recently been in the headlines, with claims that he has donated $10 million to an organization known as the International Visegrad Fund to support Ukraine. Furthermore, some reports have suggested that DiCaprio's grandmother came from Ukraine, and the donation was therefore personally motivated. The International Visegrad Fund is an international donor organization formed by the Visegrad Group countries \u2013 Czechia, Hungary, Poland, and Slovakia \u2013 to promote regional cooperation in the Visegrad region (V4). When checking the personal website of DiCaprio as well as the International Visegrad Fund, we could not find any reports or articles referring to donations in support of Ukraine. Additionally, on March 8, the International Visegrad Fund published a denial on Facebook by citing the matter as an example: \"some media outlets informed today that the International Visegrad Fund announced a USD 10 million donation from a famous American actor to help Ukraine. \" Further, they advised that \"Nowadays, fake news is present in any form and shape. Please, be aware of your sources, follow credible journalists and media, and do not spread fake news. \" On March 9, Insider cited a person close to DiCaprio claiming he had donated to the humanitarian groups supporting Ukraine such as CARE, International Rescue Committee, the United Nation's High Commissioner for Refugees, and Save the Children, but that he had no familial ties to Ukraine. Lucia Becov\u00e1, a spokesperson of the International Visegrad Fund, told Insider that \"We are not aware, neither do we have any information about Leonardo DiCaprio donating USD 10 million/\u00a37. 6 million to Ukraine. International Visegrad Fund has not announced this information. \" Finally, since DiCaprio is a messenger for peace for the UNHCR, he donates to humanitarian organizations but no figures are disclosed publically, and we have found no evidence for the claim he has donated to the International Visegrad Fund.
{"label": false}
"""  # noqa: E501

FACT_CHECK_USER_PROMPT = """Given the claim and evidence provided, classify the claim as {{"label": true}} if it is true, and {{"label": false}} if it is false.
    {examples}
    Claim: {claim}
    Evidence: {evidence}
"""  # noqa: E501

NUMERATION_NEGATIVE_EXAMPLES_SYNTHETIC = """
The numbers in the evidence may or may not match the claim. For example:
Claim: The Eiffel Tower is three hundred and fifty-one meters tall.
Evidence: The Eiffel Tower is 330 meters tall.
{"label": false}

Claim: The Eiffel Tower is three hundred and fifty-one meters tall.
Evidence: The Eiffel Tower is 351 meters tall.
{"label": true}
"""

APPROXIMATION_NEGATIVE_EXAMPLES_SYNTHETIC = """
The numbers in the evidence may or may not match the claim. For example:
Claim: The population of Canada in 2023 was about 45 million.
Evidence: The population of Canada in 2023 was 40.5 million by October 2023.
{"label": false}

Claim: The population of Canada in 2023 was about 40 million.
Evidence: The population of Canada in 2023 was 40.5 million by October 2023.
{"label": true}
"""

NEGATION_NEGATIVE_EXAMPLES_SYNTHETIC = """
The numbers in the evidence may or may not match the claim. For example:
Claim: The year-over-year U.S. inflation rate at the end of 2024 was -2.9%
Evidence: The year-over-year U.S. inflation rate at the end of 2024 was 2.9%.
{"label": false}

Claim: The year-over-year U.S. inflation rate at the end of 2024 was -2.9%
Evidence: The year-over-year U.S. inflation rate at the end of 2024 was 2.9%.
{"label": false}
"""

RANGE_NEGATIVE_EXAMPLES_SYNTHETIC = """
The numbers in the evidence may or may not match the claim. For example:
Claim: The birth rate in Japan in 2023 was between 2 to 2.5.
Evidence: The birth rate in Japan in 2023 was 1.2.
{"label": false}

Claim: The birth rate in Japan in 2023 was between 1 to 1.5.
Evidence: The birth rate in Japan in 2023 was 1.2.
{"label": true}
"""

# # Does not include True example
RANDOM_NEGATIVE_EXAMPLES_SYNTHETIC = """
The numbers in the evidence may or may not match the claim. For example:
Claim: Saturn has 789 moons.
Evidence: Discoveries bring Saturn's total moon count to 274, nearly triple Jupiter's and more than the total number of known moons around the other planets.
{"label": false}
"""

# Does not include True example
MASKING_NEGATIVE_EXAMPLES_SYNTHETIC = """
The numbers in the evidence may or may not match the claim. For example:
Claim: The Wembley Stadium in London has a seating capacity of ######.
Evidence: The Wembley Stadium in London has a seating capacity of 90,000.
{"label": false}
"""


NEGATIVE_EXAMPLES_SYNTHETIC_ALL_COMBINED = """
The numbers in the evidence may not match the claim. For example:
Claim: The Eiffel Tower is three hundred and fifty-one meters tall.
Evidence: The Eiffel Tower is 330 meters tall.
{"label": false}

Claim: The year-over-year U.S. inflation rate at the end of 2024 was -2.9%
Evidence: The year-over-year U.S. inflation rate at the end of 2024 was 2.9%.
{"label": false}

Claim: The birth rate in Japan in 2023 was between 2 to 2.5.
Evidence: The birth rate in Japan in 2023 was 1.2.
{"label": false}

Claim: The population of Canada in 2023 was about 45 million.
Evidence: The population of Canada in 2023 was 40.5 million by October 2023.
{"label": false}

Claim: Saturn has 789 moons.
Evidence: Discoveries bring Saturn's total moon count to 274, nearly triple Jupiter's and more than the total number of known moons around the other planets.
{"label": false}

Claim: The Wembley Stadium in London has a seating capacity of ######.
Evidence: The Wembley Stadium in London has a seating capacity of 90,000.
{"label": false}
"""

NEGATIVE_EXAMPLES_SYNTHETIC_WITH_REASONING = """
The numbers in the evidence may not match the claim. For example:
Claim: The Eiffel Tower is three hundred and fifty-one meters tall.
Evidence: The Eiffel Tower is 330 meters tall.
Reasoning: The claim states that the Eiffel Tower is three hundred and fifty-one meters tall (351 m), but the evidence indicates it is 330 meters tall. Since the numbers do not match, the claim is false.
{"label": false}


Claim: The year-over-year U.S. inflation rate at the end of 2024 was -2.9%
Evidence: The year-over-year U.S. inflation rate at the end of 2024 was 2.9%.
Reasoning: The claim states that the inflation rate was -2.9%, but the evidence indicates it was 2.9%. The number in the claim is negative, while the number in the evidence is positive. Since the numbers do not match, the claim is false.
{"label": false}


Claim: The birth rate in Japan in 2023 was between 2 to 2.5.
Evidence: The birth rate in Japan in 2023 was 1.2.
Reasoning: The claim states that the birth rate was between 2 to 2.5, but the evidence indicates it was 1.2. The range established by the claim does not include the number in the evidence. Since the numbers do not match, the claim is false.
{"label": false}


Claim: The population of Canada in 2023 was about 45 million.
Evidence: The population of Canada in 2023 was 40.5 million by October 2023.
Reasoning: The claim states that the population was about 45 million, but the evidence indicates it was 40.5 million. The approximation in the claim is not close enough to the number in the evidence. Therefore, the claim is false.
{"label": false}


Claim: Saturn has 789 moons.
Evidence: Discoveries bring Saturn's total moon count to 274, nearly triple Jupiter's and more than the total number of known moons around the other planets.
Reasoning: The claim states that Saturn has 789 moons, but the evidence indicates it has 274 moons. Since the numbers do not match, the claim is false.
{"label": false}


Claim: The Wembley Stadium in London has a seating capacity of ######.
Evidence: The Wembley Stadium in London has a seating capacity of 90,000.
Reasoning: The claim states that the seating capacity is ######, but the evidence indicates it is 90,000. The number is not present in the claim, so we cannot verify it. Therefore, the claim is false.
{"label": false}
"""


NEGATIVE_EXAMPLES = """
Claim: 50,000 (oil spill) claims have been filed by Floridians, but only 10 percent of them have been paid.
Evidence: In her last meeting of the Florida Cabinet, state Chief Financial Officer and unsuccessful gubernatorial candidate Alex Sink took BP claims administrator Ken Feinberg to task for failing to live up to his promise to swiftly respond to oil spill claims from Floridians and Florida business owners. Sink, who has been critical of BP and the handling of the claims process, said Floridians are having their claims denied without being given a reason, that people in similar situations are having their claims processed differently, and that Feinberg has been \"incredibly unresponsive to Floridians and to Florida business owners. \"It just reveals that this process continues to be inefficient,\" Sink said near the end of a five-hour Cabinet meeting on Dec. 7, 2010. \"One hundred, fifty thousand claims have been filed by Floridians, but only 40 percent of them have been paid, a third of them have been denied. and then we have 20 percent that are still in limbo land. \"Now, more than seven months after the initial Deepwater Horizon oil rig explosion, we wanted to see if Sink is describing the speed of the claims process correctly. To recap, President Barack Obama selected Feinberg to manage the disbursement of $20 billion provided by BP to compensate spill victims through something called the Gulf Coast Claims Facility. Feinberg has been criticized by politicians at the state and national level for months for what some politicians have called a much too slow response. Sink spokesman Kevin Cate pointed us to the Gulf Coast Claims Facility's own analysis as evidence of the slow claims process. Cate used the latest statistics available for Dec. 7. We should note that as of Dec. 7, Feinberg has awarded more than $900 million to Floridians and Florida business owners. Almost all of the money paid out is compensation for lost earnings or profits. You can read the latest report for yourself here. In lashing out against BP claims administrator Feinberg, Sink noted that of all claims made in Florida following the Gulf of Mexico oil spill, only 40 percent had been paid. She's just about right on. So far, about 43 percent of claims submitted by Floridians or Florida business owners have been paid, another 39 percent have been denied and a little more than 18 percent are still under some type of review.
{"label": false}
"""  # noqa: E501

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "boolean"}},
    "required": ["label"],
}


class PromptType(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    FEW_SHOT_WITH_NEGATIVE = "few_shot_with_negative"
    NEGATIVE_SYNTHETIC_ALL = "negative_synthetic_all"
    NEGATIVE_SYNTHETIC_ALL_WITH_REASONING = (
        "negative_synthetic_all_with_reasoning"
    )
    PERTURB_APPROX = "approximation"
    PERTURB_RANGE = "range"
    PERTURB_NUMERATION = "numeration"
    PERTURB_NEGATION = "negative_number"
    PERTURB_RANDOM = "random_replacement"
    PERTURB_MASKING = "masking"


def get_output_schema() -> dict:
    """Returns the output schema for the fact-checking task.

    Returns:
        The output schema as a dictionary.
    """
    return OUTPUT_SCHEMA


def get_fact_check_system_prompt() -> str:
    """Returns the system prompt for fact-checking.

    Returns:
        The system prompt for fact-checking.
    """
    return FACT_CHECK_SYSTEM_PROMPT


prompt_map = {
    PromptType.ZERO_SHOT: "",
    PromptType.FEW_SHOT: FEW_SHOT_EXAMPLES,
    PromptType.FEW_SHOT_WITH_NEGATIVE: FEW_SHOT_EXAMPLES + NEGATIVE_EXAMPLES,
    PromptType.NEGATIVE_SYNTHETIC_ALL: NEGATIVE_EXAMPLES_SYNTHETIC_ALL_COMBINED,
    PromptType.NEGATIVE_SYNTHETIC_ALL_WITH_REASONING: NEGATIVE_EXAMPLES_SYNTHETIC_WITH_REASONING,
    PromptType.PERTURB_APPROX: APPROXIMATION_NEGATIVE_EXAMPLES_SYNTHETIC,
    PromptType.PERTURB_RANGE: RANGE_NEGATIVE_EXAMPLES_SYNTHETIC,
    PromptType.PERTURB_NUMERATION: NUMERATION_NEGATIVE_EXAMPLES_SYNTHETIC,
    PromptType.PERTURB_NEGATION: NEGATION_NEGATIVE_EXAMPLES_SYNTHETIC,
    PromptType.PERTURB_RANDOM: RANDOM_NEGATIVE_EXAMPLES_SYNTHETIC,
    PromptType.PERTURB_MASKING: MASKING_NEGATIVE_EXAMPLES_SYNTHETIC,
}


def get_fact_check_user_prompt(
    claim: str,
    evidence: str,
    prompt_type: PromptType = PromptType.FEW_SHOT,
) -> str:
    """Returns the user prompt for fact-checking with claim and evidence.

    Args:
        claim: The claim to be fact-checked.
        evidence: The evidence to support the claim.
        prompt_type: The type of examples to include in the prompt.

    Returns:
        The formatted user prompt.
    """
    examples = prompt_map[prompt_type]
    return FACT_CHECK_USER_PROMPT.format(
        claim=claim, evidence=evidence, examples=examples
    )
