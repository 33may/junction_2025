import os, json
import random
import re
import time

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, RootModel
from typing import List, Literal

load_dotenv()

key = os.getenv("OPENAI_KEY")


def generate_personas(n):

    client = OpenAI(api_key=key)


    # --- Enums ---
    Sex = Literal["male","female","transgeneder","non_binary","intersex"]
    Race = Literal["white","black_or_african","asian","south_asian","east_asian","middle_eastern","latino_hispanic","indigenous","pacific_islander","mixed","other","prefer_not_to_say"]
    Country = Literal["US","CA","MX","GB","IE","FR","DE","NL","BE","LU","ES","PT","IT","CH","AT","PL","CZ","SK","HU","RO","BG","GR","SE","NO","DK","FI","EE","LV","LT","SI","HR","RU","UA","TR","BR","AR","CL","CN","JP","KR","IN","ID","AU","NZ","ZA","EG","NG","KE"]
    PoliticalViews = Literal["very_conservative","conservative","slightly_conservative","moderate","slightly_liberal","liberal","very_liberal","other"]
    PartyId = Literal["republican","democrat","independent_close_to_republican","independent_close_to_democrat","independent_pure","green","libertarian","other","none"]
    ResidenceAt16 = Literal["northeast","midwest","south","west","west_south_central","east_south_central","mountain","pacific","europe","other"]
    SameResidenceSince16 = Literal["same_house","same_city","different_state","different_country"]
    FamilyStructureAt16 = Literal["lived_with_parents","single_parent","guardian","other"]
    FamilyIncomeAt16 = Literal["low","average","high","prefer_not_to_say"]
    Degree = Literal["no_schooling","primary","some_high_school","high_school","some_college","associate","bachelor","master","doctorate","other"]
    MothersWorkHistory = Literal["yes","no","part_time","unknown"]
    MaritalStatus = Literal["never_married","married","cohabiting","separated","divorced","widowed"]
    WorkStatus = Literal["employed","with_a_job_but_not_at_work","unemployed_looking","unemployed_not_looking","student","retired","homemaker","disabled","living_on_financial_support","other"]
    MilitaryServiceDuration = Literal["no_active_duty","less_than_1_year","1_3_years","4_10_years","more_than_10_years","currently_serving"]
    Religion = Literal["none","catholic","protestant","orthodox","muslim","jewish","hindu","buddhist","other","prefer_not_to_say"]
    YesNo = Literal["yes","no"]
    Citizenship = Literal["citizen","permanent_resident","temporary_resident","visa_holder","undocumented","other"]
    Wealth = Literal["less_than_5k","5k_25k","25k_100k","100k_500k","more_than_500k","prefer_not_to_say"]
    LowMedHigh = Literal["low","medium","high"]

    # --- Nested objects ---
    class PersonalityTraits(BaseModel):
        extroversion: LowMedHigh
        openness_to_experience: LowMedHigh
        conscientiousness: LowMedHigh
        agreeableness: LowMedHigh
        emotional_stability: LowMedHigh
        aggressiveness: LowMedHigh

    class SkillsAndCapabilities(BaseModel):
        debate_ability: LowMedHigh
        critical_thinking: LowMedHigh
        contribute_own_ideas: LowMedHigh
        leadership: LowMedHigh
        resilience_under_pressure: LowMedHigh
        teamwork: LowMedHigh
        creativity: LowMedHigh

    class RaceOptions(BaseModel):
        white: Literal["european","german","irish","italian","slavic","scandinavian","north_american", "african_american","nigerian","ethiopian","jamaican", "chinese","japanese","korean","vietnamese","filipino","thai","indian","pakistani","bangladeshi","arab","turkish","israeli_jewish","armenian", "mexican","chilean", "native_american","aboriginal_australian","mixed_white_black","mixed_white_asian","mixed_white_latino","mixed_black_asian"]

    # --- Main profile ---
    class SurveyProfile(BaseModel):
        first_name: str
        last_name: str
        age: int  # ask model to keep 16–90
        sex: Sex
        race: Race
        race_options: RaceOptions
        country: Country
        city_state: str = Field(alias="city/state")
        political_views: PoliticalViews
        party_identification: PartyId
        # residence_at_16: ResidenceAt16
        # same_residence_since_16: SameResidenceSince16
        family_structure_at_16: FamilyStructureAt16
        family_income_at_16: FamilyIncomeAt16
        fathers_highest_degree: Degree
        mothers_highest_degree: Degree
        # mothers_work_history: MothersWorkHistory
        marital_status: MaritalStatus
        work_status: WorkStatus
        military_service_duration: MilitaryServiceDuration
        religion: Religion
        # religion_at_16: Religion
        immigrated_to_current_country: YesNo
        citizenship_status: Citizenship
        highest_degree_received: Degree
        speak_other_language: YesNo
        total_wealth: Wealth
        personality_traits: PersonalityTraits
        skills_and_capabilities: SkillsAndCapabilities
        story: str

    # --- Wrapper object required by Structured Outputs (top-level must be an object) ---
    class PersonasPayload(BaseModel):
        personas: List[SurveyProfile]

    # --- Ask for N personas inside the object personas=[...] ---
    n = 9

    # prompt = f"""
    # Generate EXACTLY {n} Really different(country, education level, ethnicity, race, income) personas as an object, the persons dont need to be strictly positive, consider average level of intelligence, income and life stability and make proportion of personas somehow relevant to distribution in real world. Ensure to have people with different political belives also including marginals(far right, far left, authoritan, comunist):
    # {{
    #   "personas": [ ... ]
    # }}
    # Each item must match the provided Pydantic model (enums exactly). Keep relations reasonable:
    # - age in [16, 90]; work_status consistent with age; education consistent with age.
    # - country aligns with city/state formatting and party_identification.
    # - immigration and citizenship_status should make sense together.
    # - total_wealth aligns with age, work_status, and the story.
    # - story (100–200 words) must be coherent with all fields and reflect personality_traits.
    # Return ONLY the JSON object with the 'personas' array. Add more information related to social view to make it helpful in reasoning and providing diverse group opinion in extremism and hate speach discussions.
    # """

    prompt = f"""
    Generate EXACTLY {n} Really different(country, education level, ethnicity, race, income) personas as an object, the persons dont need to 
    be strictly positive, consider average level of intelligence, income and life stability and make proportion of personas somehow relevant 
    to distribution in real world. Ensure that each person has a different political belief, specifcially moderate far-right, moderate far-left, 
    centrist, moderate authoritarian, moderate libertarian, far-left authoritarian, far-right libertarian, far-left libertarian, 
    far-right authoritarian:
    {{
      "personas": [ ... ]
    }}
    Each item must match the provided Pydantic model (enums exactly). Keep relations reasonable:
    - age in [16, 90]; work_status consistent with age; education consistent with age.
    - country aligns with city/state formatting and party_identification.
    - immigration and citizenship_status should make sense together.
    - total_wealth aligns with age, work_status, and the story.
    - story (100–200 words) must be coherent with all fields and reflect personality_traits.
    Return ONLY the JSON object with the 'personas' array. Add more information related to social view to make it helpful in reasoning and providing diverse group opinion in extremism and hate speach discussions.
    """

    start = time.time()

    response = client.responses.parse(
        # model="gpt-5-nano",
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": "Return only data that parses into PersonasPayload. No explanations."},
            {"role": "user", "content": prompt.strip()},
        ],
        text_format=PersonasPayload,
    )

    payload = response.output_parsed
    json_payload = json.dumps(payload.model_dump(by_alias=True), indent=2, ensure_ascii=False)

    # Save to file
    if not os.path.exists("./data/personas.json"):
        with open("./data/personas.json", "w", encoding="utf-8") as f:
            f.write(json_payload)
    else:
        # Detects the next available index.
        name_pattern = re.compile(r"personas_(\d+)\.json$")
        existing_files = [s for s in os.listdir("./data") if name_pattern.match(s)]
        
        if existing_files:
            existing_indices = [int(name_pattern.match(s).group(1)) for s in existing_files]
            next_index = max(existing_indices) + 1
        else :
            next_index = 1

        # Writes to file.
        with open(f"./data/personas_{next_index}.json", "w", encoding="utf-8") as f:
            f.write(json_payload)

    print(json_payload)

    print(f"Completed in {time.time() - start} seconds.")

    return json_payload


def get_set_of_personas(n, personas_path = "../data/personas.json"):
    with open(personas_path) as f:
        personas = json.load(f)
    personas = personas["personas"]
    k = min(n, len(personas))
    return random.sample(personas, k)


#Test Run
# generate_personas(9)