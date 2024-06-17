from openai import OpenAI
import gradio as gr

import os
import json
import re
from openai import OpenAI
import logging
import uuid
import random
import time
import glob
import argparse
import streamlit as st
import pandas as pd



class TransChat:
    """
    TransChat is a class that handles the translation life cycle of a book.
    It involves the interaction between agents.
    """
    def __init__(
        self, client, src_lang, tgt_lang, text_path, save_dir,
        num_senior_editors=2, 
        num_junior_editors=2,
        num_translators=2, 
        num_localization_specialists=2, 
        num_proofreaders=2,
        num_beta_readers=2,
        max_turns=3,
        max_retry=3,
        max_rerun=5,
    ):

        self.client = client
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.text = self.read_text(text_path)
        self.book = self.split_chapter(self.text)
        self.book_summary = None
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.project_save_dir = os.path.join(save_dir, os.path.basename(text_path))
        os.makedirs(self.project_save_dir, exist_ok=True)
        self.num_senior_editors = num_senior_editors
        self.num_junior_editors = num_junior_editors
        self.num_translators = num_translators
        self.num_localization_specialists = num_localization_specialists
        self.num_proofreaders = num_proofreaders
        self.num_beta_readers = num_beta_readers
        self.max_turns = max_turns
        self.max_retry = max_retry
        self.max_rerun = max_rerun
        self.total_cost = 0

        self.model = "gpt-4-1106-preview"
        self.input_rate = 0.00001
        self.output_rate = 0.00003
        self.conversations = []

        # self.model = "gpt-3.5-turbo-1106"
        # self.input_rate = 0.000001
        # self.output_rate = 0.000002

        self.glossary = []

        self.company_prompt = f"TransChat is a {self.src_lang} translation firm specializing in translating books across a wide range of languages. It utilizes a team with diverse backgrounds that include roles such as Senior Editor, Junior Editor, Translator, and more. Its goal is to connect cultures and languages through precise, engaging, and culturally respectful literature translations, thereby promoting a worldwide community united by the art of storytelling."
        self.project_members = {
            "ceo": {
                "role_prompt": "You are a proficient CEO, consistently adept at precisely grasping your clients' needs. You are designed to output JSON.",
                "model": self.model,
                "input_rate": self.input_rate, 
                "output_rate": self.output_rate,
            },
        }
        self.senior_editor_pool = []
        self.junior_editor_pool = []
        self.translator_pool = []
        self.localization_specialist_pool = []
        self.proofreader_pool = []

        print(self.company_prompt)

        self.curr_rerun = 0


    def read_text(self, path):
        """
        :param path: path to the text file
        :return: a list of sentences
        """
        lst = []
        with open(path, "r") as f:
            for line in f.readlines():
                if line.strip() != "":
                    lst.append(line)
        return lst

    def write_text(self, path, text):
        """
        :param path: path to the text file
        :param text: text to be written
        """
        with open(path, "w") as f:
            f.write(text+"\n")

    def write_jsonl(self, path, data):
        """
        :param path: path to the jsonl file
        :param data: data to be written
        """
        self.curr_rerun = 0
        print(f"Writing the data to {path}...")
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False)+"\n")

    def read_jsonl(self, path):
        """
        :param path: path to the jsonl file
        :return: a list of json objects
        """
        lst = []
        with open(path, "r") as f:
            for line in f.readlines():
                lst.append(json.loads(line))
        return lst

    def split_chapter(self, text):
        """
        :param text: text to be splitted
        :return: a list of sentences
        """
        print("Splitting the text into chapters...")
        book = []
        chapter = []
        for l in text:
            
            if bool(re.search(r"[\u4e00-\u9fa5\d]+章\s", l)) and len(chapter) > 0:
                dic = {
                    "chapter_title": chapter[0].strip(),
                    "chapter_text": "\n".join(chapter),
                }
                book.append(dic)
                chapter = [l.strip()]
            else:
                chapter.append(l.strip())

        dic = {
            "chapter_title": chapter[0].strip(),
            "chapter_text": "\n".join(chapter),
            # "chapter_summary": "",
        }
        book.append(dic)
        return book
    
    def write_conversations(self, assistant, message):
        dialogue_turn = """###assistant###:###message###"""
        st.write(dialogue_turn.replace("###assistant###", assistant).replace("###message###", message))
        return 

    def compute_cost(self, prev_messages):
        """
        compute the cost of the conversation
        """
        costs = {}
        for m in prev_messages:
            costs[m["uuid"]] = m["cost"]
        return sum(costs.values())

    def evaluate_translation(self, chapter_text, chapter_translation):
        prev_messages = []
        translation_guildelines = self.translation_guidelines
        message = f"Translation Guidelines:\n\n{translation_guildelines}\n\nChapter Text:\n\n{chapter_text}\n\nChapter Translation:\n\n{chapter_translation}\n\nConsiderring the translation guidelines, including the glossary, book summary, tone, style, and target audience, please carefully evaluate the translation and provide a detailed justification. Ensure that the translation aligns with the original chapter text closely."
        prev_messages.append({"role": "junion_editor", "content": message})
        st.chat_message(self.project_roles["junion_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"finalize\": bool}. The value of \"finalize\" should be set to true if the translation is of high quality and does not require any further editing. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=message,
            content_key="finalize",
            additional_system_message=additional_system_message,
            prev_messages=[],
        )
        print(content)
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        return content, prev_messages

    def execute(self):

        
        self.initialize_company()
        st.chat_message("sys").write(self.company_prompt + "\n Our employees are:")
        all_members = self.senior_editor_pool + self.junior_editor_pool +self.translator_pool +self.localization_specialist_pool +self.proofreader_pool
        df = pd.DataFrame().from_dict(
            {"Members":[elem["name"] for elem in all_members],"Profile":[elem["text"][8:] for elem in all_members]}
        )
        st.chat_message("sys").table(df)

       
        self.initialize_project()
        # project_members = list(self.project_members.items())
        st.chat_message("sys").write(f"The project is to translate a book from {self.src_lang} to {self.tgt_lang}, which has {len(self.book)} chapters and {len(self.text)} sentences. The project team is:")
        df = pd.DataFrame().from_dict(
            {"Members":list(self.project_members.keys()),"Profile":[elem["role_prompt"][8:] for elem in list(self.project_members.values())]}
        )
        self.project_roles = {name: profile["role_prompt"].split(",")[8:] for name, profile in self.project_members.items()}
        st.chat_message("sys").table(df)


        st.chat_message("sys").write("Preparing the project")
        self.prepare()


        self.translate()



        self.post_process()

    def initialize_company(self):
        """
        initialize the company
        """
        print("*********************************************************************")
        print("********************** Initializing the company... ******************")
        print("*********************************************************************")
        print(self.company_prompt)

        # st.chat_message("sys").write("********************** Initializing the Company ******************")
        



        company_dir = os.path.join(self.save_dir, "company")
        os.makedirs(company_dir, exist_ok=True)
        
        self.initialize_senior_editor_pool(company_dir)
        self.initialize_junior_editor_pool(company_dir)
        self.initialize_translator_pool(company_dir)
        self.initialize_localization_specialist_pool(company_dir)
        self.initialize_proofreader_pool(company_dir)


    def initialize_senior_editor_pool(self, company_dir):
        """
        Initialize the senior editor pool
        """
        print("Initializing the senior editor pool...")
        
        senior_editor_path = os.path.join(company_dir, "senior_editor_pool.jsonl")
        if os.path.exists(senior_editor_path):
            print(f"Loading the senior editor pool from {senior_editor_path}...")
            self.senior_editor_pool = self.read_jsonl(senior_editor_path)
            return

        senior_editors = []
        idx = 0
        while idx < self.num_senior_editors:
            print(f"Initializing senior editor {idx}...")
            names = ", ".join([e["name"] for e in senior_editors])
            message = f"Existing Senior Editors: {names}\n\nGenerate a new, fictional profile for a senior editor. This should include their name, the languages they speak, nationality, gender, age, educational background, personality traits, hobbies, and rate of pay per word. Also, include their experience, measured in years of work. Ensure that the information provided is highly diverse, reflecting a wide range of backgrounds, personality traits, and experiences, including negative ones."
            additional_system_message = "Your response should always be in JSON format as follows: {\"profile\": {\"name\": string, \"languages\": [string], \"nationality\": string, \"gender\": string, \"age\": int, \"education\": string, \"personality\": [string], \"hobbies\": [string], \"rate_per_word\": float, \"years_of_working\": int}}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="profile",
                additional_system_message=additional_system_message,
                prev_messages=[],
            )
            profile = content["profile"]
            profile["profession"] = "senior_editor"
            profile["uuid"] = str(uuid.uuid4())
            if profile["name"] in [e["name"] for e in senior_editors]:
                continue

            message = f"{json.dumps(profile)}\nPlease write a paragraph based on the provided information, starting with \"You are {profile['name']}\". Note that the professon must be included in the paragraph."
            additional_system_message = "Your response should always be in JSON format as follows: {\"text\": string}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="text",
                additional_system_message=additional_system_message,
                prev_messages=[]
            )
            profile["text"] = content["text"]
            print(profile)
            senior_editors.append(profile)
            idx += 1
            name, text = profile["name"], profile["text"]
            st.chat_message("ceo").write(f"recuiting senior editors: {text[8:]}")

        self.senior_editor_pool = senior_editors
        self.write_jsonl(senior_editor_path, senior_editors)

    def initialize_junior_editor_pool(self, company_dir):
        """
        Initialize the junior editor pool
        """
        print("Initializing the junior editor pool...")
        junior_editor_path = os.path.join(company_dir, "junior_editor_pool.jsonl")
        if os.path.exists(junior_editor_path):
            print(f"Loading the junior editor pool from {junior_editor_path}...")
            self.junior_editor_pool = self.read_jsonl(junior_editor_path)
            return

        junior_editors = []
        responses = []
        idx = 0
        while idx < self.num_junior_editors:
            print(f"Initializing junior editor {idx}...")
            names = ", ".join([e["name"] for e in junior_editors])
            message = f"Existing Junior Editors: {names}\n\nGenerate a new, fictional profile for a junior editor. This should include their name, the languages they speak, nationality, gender, age, educational background, personality traits, hobbies, and rate of pay per word. Also, include their experience, measured in years of work. Ensure that the information provided is highly diverse, reflecting a wide range of backgrounds and experiences."
            additional_system_message = "Your response should always be in JSON format as follows: {\"profile\": {\"name\": string, \"languages\": [string], \"nationality\": string, \"gender\": string, \"age\": int, \"education\": string, \"personality\": [string], \"hobbies\": [string], \"rate_per_word\": float, \"years_of_working\": int}}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="profile",
                additional_system_message=additional_system_message,
                prev_messages=[],
            )
            profile = content["profile"]
            profile["profession"] = "junior_editor"
            if profile["name"] in [e["name"] for e in junior_editors]:
                continue
            
            message = f"{json.dumps(profile)}\nPlease write a paragraph based on the provided information, starting with \"You are {profile['name']}\". Note that the professon must be included in the paragraph."
            additional_system_message = "Your response should always be in JSON format as follows: {\"text\": string}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="text",
                additional_system_message=additional_system_message,
                prev_messages=[]
            )
            profile["text"] = content["text"]
            profile["uuid"] = str(uuid.uuid4())
            junior_editors.append(profile)
            idx += 1
            name, text = profile["name"], profile["text"]
            st.chat_message("ceo").write(f"recuiting junior editors: {text[8:]}")

        self.junior_editor_pool = junior_editors
        self.write_jsonl(junior_editor_path, junior_editors)

    def initialize_translator_pool(self, company_dir):
        """
        Initialize the translator pool
        """
        print("Initializing the translator pool...")
        translator_path = os.path.join(company_dir, "translator_pool.jsonl")
        if os.path.exists(translator_path):
            print(f"Loading the translator pool from {translator_path}...")
            self.translator_pool = self.read_jsonl(translator_path)
            return

        translators = []
        idx = 0
        while idx < self.num_translators:
            print(f"Initializing translator {idx}...")
            names = ", ".join([e["name"] for e in translators])
            message = "Existing Translators: {names}\n\nGenerate a new, fictional profile for a translator. This should include their name, the languages they speak, nationality, gender, age, educational background, personality traits, hobbies, and rate of pay per word. Also, include their experience, measured in years of work. Ensure that the information provided is highly diverse, reflecting a wide range of backgrounds and experiences."
            additional_system_message = "Your response should always be in JSON format as follows: {\"profile\": {\"name\": string, \"languages\": [string], \"nationality\": string, \"gender\": string, \"age\": int, \"education\": string, \"personality\": [string], \"hobbies\": [string], \"rate_per_word\": float, \"years_of_working\": int}}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="profile",
                additional_system_message=additional_system_message,
                prev_messages=[],
            )
            profile = content["profile"]
            profile["profession"] = "translator"
            if profile["name"] in [e["name"] for e in translators]:
                continue

            message = f"{json.dumps(profile)}\nPlease write a paragraph based on the provided information, starting with \"You are {profile['name']}\". Note that the professon must be included in the paragraph."
            additional_system_message = "Your response should always be in JSON format as follows: {\"text\": string}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="text",
                additional_system_message=additional_system_message,
                prev_messages=[]
            )
            profile["text"] = content["text"]
            profile["uuid"] = str(uuid.uuid4())
            translators.append(profile)
            idx += 1
            name, text = profile["name"], profile["text"]
            st.chat_message("ceo").write(f"recuiting translators: {text[8:]}")
        
        self.translator_pool = translators
        self.write_jsonl(translator_path, translators)

    def initialize_localization_specialist_pool(self, company_dir):
        """
        Initialize the localization specialist pool
        """
        print("Initializing the localization specialist pool...")
        localization_specialist_path = os.path.join(company_dir, "localization_specialist_pool.jsonl")
        if os.path.exists(localization_specialist_path):
            print(f"Loading the localization specialist pool from {localization_specialist_path}...")
            self.localization_specialist_pool = self.read_jsonl(localization_specialist_path)
            return
        
        localization_specialists = []
        idx = 0
        while idx < self.num_localization_specialists:
            print(f"Initializing localization specialist {idx}...")
            names = ", ".join([e["name"] for e in localization_specialists])
            message = f"Existing Localization Specialists: {names}\n\nGenerate a new, fictional profile for a localization specialist. This should include their name, the languages they speak, nationality, gender, age, educational background, personality traits, hobbies, and rate of pay per word. Also, include their experience, measured in years of work. Ensure that the information provided is highly diverse, reflecting a wide range of backgrounds and experiences."
            additional_system_message = "Your response should always be in JSON format as follows: {\"profile\": {\"name\": string, \"languages\": [string], \"nationality\": string, \"gender\": string, \"age\": int, \"education\": string, \"personality\": [string], \"hobbies\": [string], \"rate_per_word\": float, \"years_of_working\": int}}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="profile",
                additional_system_message=additional_system_message,
                prev_messages=[],
            )
            profile = content["profile"]
            profile["profession"] = "localization_specialist"
            if profile["name"] in [e["name"] for e in localization_specialists]:
                continue

            message = f"{json.dumps(profile)}\nPlease write a paragraph based on the provided information, starting with \"You are {profile['name']}\". Note that the professon must be included in the paragraph."
            additional_system_message = "Your response should always be in JSON format as follows: {\"text\": string}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="text",
                additional_system_message=additional_system_message,
                prev_messages=[]
            )
            profile["text"] = content["text"]
            profile["uuid"] = str(uuid.uuid4())
            # print(profile)
            localization_specialists.append(profile)
            idx += 1
            name, text = profile["name"], profile["text"]

            st.chat_message("ceo").write(f"Recuiting localization specialists: {text[8:]}")

        self.localization_specialist_pool = localization_specialists
        self.write_jsonl(localization_specialist_path, localization_specialists)

    def initialize_proofreader_pool(self, company_dir):
        """
        Initialize the proofreader pool
        """
        print("Initializing the proofreader pool...")
        proofreader_path = os.path.join(company_dir, "proofreader_pool.jsonl")
        if os.path.exists(proofreader_path):
            print(f"Loading the proofreader pool from {proofreader_path}...")
            self.proofreader_pool = self.read_jsonl(proofreader_path)
            return

        proofreaders = []
        idx = 0
        while idx < self.num_proofreaders:
            print(f"Initializing proofreader {idx}...")
            names = ", ".join([e["name"] for e in proofreaders])
            message = f"Existing Proofreaders: {names}\n\nGenerate a new, fictional profile for a proofreader. This should include their name, the languages they speak, nationality, gender, age, educational background, personality traits, hobbies, and rate of pay per word. Also, include their experience, measured in years of work. Ensure that the information provided is highly diverse, reflecting a wide range of backgrounds and experiences."
            additional_system_message = "Your response should always be in JSON format as follows: {\"profile\": {\"name\": string, \"languages\": [string], \"nationality\": string, \"gender\": string, \"age\": int, \"education\": string, \"personality\": [string], \"hobbies\": [string], \"rate_per_word\": float, \"years_of_working\": int}}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="profile",
                additional_system_message=additional_system_message,
                prev_messages=[],
            )
            profile = content["profile"]
            profile["profession"] = "proofreader"
            if profile["name"] in [e["name"] for e in proofreaders]:
                continue

            message = f"{json.dumps(profile)}\nPlease write a paragraph based on the provided information, starting with \"You are {profile['name']}\". Note that the professon must be included in the paragraph."
            additional_system_message = "Your response should always be in JSON format as follows: {\"text\": string}. Please do not change the key of the JSON object."
            content, response = self.call_api(
                assistant="ceo",
                message=message,
                content_key="text",
                additional_system_message=additional_system_message,
                prev_messages=[]
            )
            profile["text"] = content["text"]
            profile["uuid"] = str(uuid.uuid4())
            # print(profile)
            proofreaders.append(profile)
            idx += 1
            name, text = profile["name"], profile["text"]
            st.chat_message("ceo").write(f"Recuiting proofreaders: {text[8:]}")

        self.proofreader_pool = proofreaders
        self.write_jsonl(proofreader_path, proofreaders)

    def initialize_project(self):
        """
        initialize the project
        """
        print("*********************************************************************")
        print("********************** Initializing the project... ******************")
        print("*********************************************************************")
        print(f"The project is to translate a book from {self.src_lang} to {self.tgt_lang}.")
        print(f"The book has {len(self.book)} chapters.")
        print(f"The book has {len(self.text)} sentences.")
        project_members_path = os.path.join(self.project_save_dir, "project_members.jsonl")
        if os.path.exists(project_members_path):
            print(f"Loading the project members from {project_members_path}...")
            self.project_members = self.read_jsonl(project_members_path)[0]

            print("Project members:")
            for k, v in self.project_members.items():
                # v["model"] = self.model
                # v["input_rate"] = self.input_rate
                # v["output_rate"] = self.output_rate
                print(f"{k}: ")
                print(f"Role Prompt: {v['role_prompt']}")
                print(f"Model: {v['model']}")

            return

        self.assign_project_to_role("ceo", "senior_editor")
        self.assign_project_to_role("senior_editor", "junior_editor")
        self.assign_project_to_role("senior_editor", "translator")
        self.assign_project_to_role("senior_editor", "localization_specialist")
        self.assign_project_to_role("senior_editor", "proofreader")

        self.write_jsonl(project_members_path, [self.project_members])
        print("here")

    def assign_project_to_role(self, assignor, assignee):
        """
        assign a role to the project
        """

        print(f"Assigning a {assignee} to the project...")

        prev_messages = []

        assignee_pool_map = {
            "senior_editor": self.senior_editor_pool,
            "junior_editor": self.junior_editor_pool,
            "translator": self.translator_pool,
            "localization_specialist": self.localization_specialist_pool,
            "proofreader": self.proofreader_pool,
        }
        assignee_title_map = {
            "senior_editor": "Senior Editor",
            "junior_editor": "Junior Editor",
            "translator": "Translator",
            "localization_specialist": "Localization Specialist",
            "proofreader": "Proofreader",
        }

        selected_assignee = None
        assignee_pool = assignee_pool_map[assignee]

        all_assignee_text = ""
        for e in assignee_pool:
            text = e["text"].replace("You are", "").strip()
            all_assignee_text += f"{text}\n----\n"

        retry = 0
        turn = 0


        st.chat_message(assignor).write(f"I need to choose a {assignee_title_map[assignee]} who fits the project best as one of my teammates")



        while retry < self.max_retry and turn < self.max_turns:
            print(f"Turn {turn}...")
            if turn == 0:
                message = f"Candidate {assignee_title_map[assignee]}s:\n{all_assignee_text}\n\nOur client would like translate a book from {self.src_lang} to {self.tgt_lang}. Based on the descriptions of the candidates, please select a {assignee_title_map[assignee]} who fits the project best as one of your teammates, providing a detailed justification."
            else:
                message = f"Based on the descriptions of the candidates, please select a {assignee_title_map[assignee]} who fits the project best as one of your teammates again, providing a detailed justification."
            additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"candidate_name\": string}. Please do not change the key of the JSON object."
            try:
                content, response = self.call_api(
                    assistant=assignor,
                    message=message,
                    content_key="candidate_name",
                    additional_system_message=additional_system_message,
                    prev_messages=prev_messages,
                )
                print(content) #TODO: printout
                assignee_name = content["candidate_name"]
                assignee_justification = content["justification"]
                assert assignee_name in [e["name"] for e in assignee_pool]
                prev_messages.append({"role": "user", "content": message})
            except Exception as e:
                print(e)
                retry += 1
                print(f"Retry {retry} times for assigning...")
                time.sleep(1)
                continue
            selected_assignee = [e for e in assignee_pool if e["name"] == assignee_name][0]
            prev_messages.append({"role": assignor, "content": assignee_justification})
            st.chat_message(assignor).write(assignee_justification)

            message = "Would you like to finalize your decision regarding this candidate, particularly in terms of their language skills?"
            additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"finalize\": bool}. Please do not change the key of the JSON object. The value of \"finalize\" should be set to true if you are satified with this candidate."
            content, response = self.call_api(
                assistant=assignor,
                message=message,
                content_key="finalize",
                additional_system_message=additional_system_message,
                prev_messages=prev_messages,
            )
            prev_messages.append({"role": "user", "content": message})
            print(content)
            finalize = content["finalize"]
            prev_messages.append({"role": assignor, "content": json.dumps(content, ensure_ascii=False)})
            st.chat_message(assignor).write(content["justification"])


            if finalize:
                break
            else:
                turn += 1

        if selected_assignee is None:
            raise Exception("Failed to assign a role to the project.")

        self.project_members[assignee] = {
            "role_prompt": selected_assignee["text"],
            "model": self.model,
            "input_rate": self.input_rate,
            "output_rate": self.output_rate,
        }
        print(f"Assigned {assignee}:")
        for k, v in selected_assignee.items():
            print(f"{k}: {v}")
        print("----------------")

        selection_path = os.path.join(self.project_save_dir, f"{assignee}_selection.jsonl")
        self.write_jsonl(selection_path, prev_messages)
             
    def prepare(self):
        """
        a sequence of steps to prepare the text for translation, including: 
        glossary translation,
        chapter summarization, 
        book summarization, 
        personnel recuitment,
        """
        self.document_glossary()
        self.summarize_chapters()
        self.summarize_book()
        self.define_guidelines()
        # self.recruit_beta_readers()
        self.finalize_preparation()

    def document_glossary(self):
        """
        document the glossary
        """
        print("*********************************************************************")
        print("********************** Documenting the glossary... ******************")
        print("*********************************************************************")
        glossary_dir = os.path.join(self.project_save_dir, "glossary")
        os.makedirs(glossary_dir, exist_ok=True)
        glossary_path = os.path.join(glossary_dir, "glossary.jsonl")
        if os.path.exists(glossary_path):
            print(f"Loading the glossary from {glossary_path}...")
            self.glossary = self.read_jsonl(glossary_path)
            return

        num_chapters = len(self.book)
        for i in range(num_chapters):
            chapter_path = os.path.join(glossary_dir, f"chapter_{i}.jsonl")
            if os.path.exists(chapter_path):
                print(f"Loading the glossary of chapter {i} from {chapter_path}...")
                self.glossary.extend(self.read_jsonl(chapter_path))
            else:
                self.document_glossary_one_chapter(i, chapter_path)

        new_glossary = []
        print(self.glossary)
        for g in self.glossary:
            if g["source"] not in [e["source"] for e in new_glossary]:
                new_glossary.append(g)

        self.glossary = new_glossary
        self.write_jsonl(glossary_path, self.glossary)

    def document_glossary_one_chapter(self, chapter_idx, save_path):
        """
        document the glossary of one chapter
        """
        print(f"Documenting the glossary of chapter {chapter_idx}...")
        prev_messages = []
        glossary = None

        curr_chapter = self.book[chapter_idx]
        chapter_title = curr_chapter["chapter_title"]
        chapter_text = curr_chapter["chapter_text"]

        glossary_text = ", ".join([e["source"] for e in self.glossary])
        message = f"Existing {self.src_lang} Glossary:\n\n{glossary_text}\n\nChapter Text:\n\n{chapter_text}\n\nPlease analyze the text and identify all specialized terms that could lead to inconsistent translations, negatively affecting the quality of the translation, such as character names and specific in-world terminologies. In your response, include only the terms in {self.src_lang} and exclude those already in the glossary. Note that the generic and non-essential terms should be excluded as well."
        # print(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"glossary\": [string]}. Please do not change the key of the JSON object."
        
        
        content, response = self.call_api(
            assistant="junior_editor",
            message=message,
            content_key="glossary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        # st.chat_message()
        prev_messages.append({"role": "senior_editor", "content": message})
        st.chat_message(self.project_roles["senior_editor"]).write(message)
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(json.dumps(content, ensure_ascii=False))

        print(prev_messages[-1])

        message = "I believe that some non-essential terms are included, while some crucial terms are omitted. In my view, the following terms could potentially lead to inconsistencies during the translation process."
        prev_messages.append({"role": "senior_editor", "content": message})
        st.chat_message(self.project_roles["senior_editor"]).write(message)
        
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"glossary\": [string]}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=None,
            content_key="glossary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(json.dumps(content, ensure_ascii=False))
        print(prev_messages[-1])

        message = f"Please review and finalize the glossary of chapter text. Please remove those generic and non-essential terms from the glossary."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"glossary\": [string]}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=None,
            content_key="glossary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(json.dumps(content, ensure_ascii=False))
        chapter_glossary = content["glossary"]
        print(content)

        chapter_glossary_pairs = self.translate_glossary(chapter_idx, save_path, chapter_glossary)
        self.write_jsonl(save_path, chapter_glossary_pairs)
        self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)
        self.glossary.extend(chapter_glossary_pairs)

    def translate_glossary(self, chapter_idx, save_path, chapter_glossary):
        """
        translate the glossary
        """
        print(f"Translating the glossary of chapter {chapter_idx}...")
        prev_messages = []
        glossary = None

        curr_chapter = self.book[chapter_idx]
        chapter_title = curr_chapter["chapter_title"]
        chapter_text = curr_chapter["chapter_text"]

        new_glossary_text = ", ".join(chapter_glossary)
        existing_glossary_text = "\n".join([json.dumps(e, ensure_ascii=False) for e in self.glossary])
        message = f"Chapter Text:\n\n{chapter_text}\n\nExisting Glossary:\n\n{existing_glossary_text}\n\nNew {self.src_lang} Glossary:\n\n{new_glossary_text}\n\nPlease translate the new glossary of chapter text from {self.src_lang} to {self.tgt_lang}. Please ensure that the new translations are consistent with the existing glossary."
        additional_system_message = additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"text\": [{\"source\": string, \"target\": string}, ...]}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=message,
            content_key="text",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": message})
        st.chat_message(self.project_roles["senior_editor"]).write(message)
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)
        # print(prev_messages[-1])

        message = f"I think the terms in the glossary can be alternatively translated as follows:"
        prev_messages.append({"role": "senior_editor", "content": message})
        st.chat_message(self.project_roles["senior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"text\": [{\"source\": string, \"target\": string}, ...]}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=None,
            content_key="text",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        print(prev_messages[-1])

        message = f"No, I disagree with you. The terms in the glossary should be translated as follows."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"text\": [{\"source\": string, \"target\": string}, ...]}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=None,
            content_key="text",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)
        
        # print(prev_messages[-1])

        message = f"I believe we've discussed this sufficiently. Please review and finalize the translations of glossary terms in chapter text, making sure to refer to the chapter's content for context. This will help ensure that each term is translated with the highest accuracy and effectiveness."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"text\": [{\"source\": string, \"target\": string}, ...]}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=None,
            content_key="text",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        chapter_glossary_pairs = content["text"]
        # print(prev_messages[-1])
        
        self.write_jsonl(save_path.replace(".jsonl", "_trans_conv.jsonl"), prev_messages)
        return chapter_glossary_pairs

    def summarize_chapters(self):
        """
        summarize each chapter
        """
        print("*********************************************************************")
        print("********************** Summarizing chapters... **********************")
        print("*********************************************************************")
        summary_dir = os.path.join(self.project_save_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        num_chapters = len(self.book)
        for i in range(num_chapters):
            chapter_path = os.path.join(summary_dir, f"chapter_{i}.jsonl")
            if os.path.exists(chapter_path):
                print(f"Loading the summary of chapter {i} from {chapter_path}...")
                self.book[i]["chapter_summary"] = self.read_jsonl(chapter_path)[0]["summary"]
            else:
                self.summarize_one_chapter(i, chapter_path)

    def summarize_one_chapter(self, chapter_idx, save_path):
        """
        summarize one chapter
        """
        print(f"Summarizing chapter {chapter_idx}...")
        prev_messages = []
        summary = None

        curr_chapter = self.book[chapter_idx]
        chapter_title = curr_chapter["chapter_title"]
        chapter_text = curr_chapter["chapter_text"]

        glossary_text = "\n".join([e["source"] + ": " + e["target"] for e in self.glossary])
        message = f"Glossary:\n\n{glossary_text}\n\nChapter Text:\n\n{chapter_text}\n\nPlease summarize the chapter text. Please ensure that the summary is consistent with the glossary."
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"summary\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=message,
            content_key="summary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)
        # print(prev_messages[-1])

        message = f"I think the chapter can be better summarized as follows:"
        prev_messages.append({"role": "senior_editor", "content": message})
        st.chat_message(self.project_roles["senior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"summary\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=None,
            content_key="summary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        # print(prev_messages[-1])

        message = f"No, I disagree with you. The chapter should be summarized as follows."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"summary\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=None,
            content_key="summary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)

        message = f"I believe we've discussed this sufficiently. Please review and finalize the summary of chapter text, making sure to refer to the chapter's content for context. This will help ensure that the summary is accurate and effective."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"summary\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=None,
            content_key="summary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)

        self.book[chapter_idx]["chapter_summary"] = content["summary"]
        self.write_jsonl(save_path, [{"summary": content["summary"]}])
        self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)

    def summarize_book(self):
        """
        summarize the book
        """
        print("*********************************************************************")
        print("********************** Summarizing the book... **********************")
        print("*********************************************************************")
        prev_messages = []
        summary = None

        summary_dir = os.path.join(self.project_save_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "book_summary.jsonl")
        if os.path.exists(summary_path):
            print(f"Loading the summary from {summary_path}...")
            self.book_summary = self.read_jsonl(summary_path)[0]["summary"]
            return

        glossary_text = "\n".join([e["source"] + ": " + e["target"] for e in self.glossary])
        chapter_summaries = "\n".join([f"Chapter {i} Summary: {self.book[i]['chapter_summary']}" for i in range(len(self.book))])

        message = f"Glossary:\n\n{glossary_text}\n\nChapter Summaries:\n\n{chapter_summaries}\n\nPlease summarize the book. Please ensure that the summary is consistent with the glossary and chapter summaries."
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"summary\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=message,
            content_key="summary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": message})
        st.chat_message(self.project_roles["senior_editor"]).write(message)
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)

        message = f"I think the book can be better summarized as follows:"
        prev_messages.append({"role": "senior_editor", "content": message})
        st.chat_message(self.project_roles["senior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"summary\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=None,
            content_key="summary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        # print(prev_messages[-1])

        message = f"No, I disagree with you. The book should be summarized as follows."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"summary\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=None,
            content_key="summary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)
        # print(prev_messages[-1])

        message = f"I believe we've discussed this sufficiently. Please review and finalize the summary of the book, making sure to refer to the summaries of each chapter. This will help ensure that the summary is accurate and effective."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"summary\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=None,
            content_key="summary",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        # print(prev_messages[-1])

        self.book_summary = content["summary"]
        self.write_jsonl(summary_path, [{"summary": content["summary"]}])
        self.write_jsonl(summary_path.replace(".jsonl", "_conv.jsonl"), prev_messages)

    def define_guidelines(self):
        """
        define the guidelines
        """
        print("*********************************************************************")
        print("********************** Defining the guidelines... *******************")
        print("*********************************************************************")
        guidelines_dir = os.path.join(self.project_save_dir, "guidelines")
        os.makedirs(guidelines_dir, exist_ok=True)

        tone_path = os.path.join(guidelines_dir, "tone.jsonl")
        if os.path.exists(tone_path):
            print(f"Loading the tone from {tone_path}...")
            self.tone = self.read_jsonl(tone_path)[0]["tone"]
        else:
            self.define_tone(tone_path)

        style_path = os.path.join(guidelines_dir, "style.jsonl")
        if os.path.exists(style_path):
            print(f"Loading the style from {style_path}...")
            self.style = self.read_jsonl(style_path)[0]["style"]
        else:
            self.define_style(style_path)

        target_audience_path = os.path.join(guidelines_dir, "target_audience.jsonl")
        if os.path.exists(target_audience_path):
            print(f"Loading the target audience from {target_audience_path}...")
            self.target_audience = self.read_jsonl(target_audience_path)[0]["target_audience"]
        else:
            self.define_target_audience(target_audience_path)

    def define_tone(self, save_path):
        """
        define the tone
        """
        print("Defining the tone...")
        prev_messages = []

        
        summary = self.book_summary
        chapter_text = self.book[0]["chapter_text"]
        message = f"Book Summary:\n\n{summary}\n\nChapter Text:\n\n{chapter_text}\n\nGiven the book summary and the sample chapter, please define the tone of the book concisely."
        additional_system_message = "Your response should always be in JSON format as follows: {\"text\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=message,
            content_key="text",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        # print(prev_messages[-1])

        self.tone = content["text"]
        self.write_jsonl(save_path, [{"tone": content["text"]}])
        self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)
    
    def define_style(self, save_path):
        """
        define the style
        """
        print("Defining the style...")
        prev_messages = []

        summary = self.book_summary
        chapter_text = self.book[0]["chapter_text"]
        message = f"Book Summary:\n\n{summary}\n\nChapter Text:\n\n{chapter_text}\n\nGiven the book summary and the sample chapter, please define the style of the book concisely."
        additional_system_message = "Your response should always be in JSON format as follows: {\"text\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=message,
            content_key="text",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        # print(prev_messages[-1])

        self.style = content["text"]
        self.write_jsonl(save_path, [{"style": content["text"]}])
        self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)

    def define_target_audience(self, save_path):
        """
        define the target audience
        """
        print("Defining the target audience...")
        prev_messages = []

        summary = self.book_summary
        chapter_text = self.book[0]["chapter_text"]
        message = f"Book Summary:\n\n{summary}\n\nChapter Text:\n\n{chapter_text}\n\nGiven the book summary and the sample chapter, please define the target audience of the book concisely."
        additional_system_message = "Your response should always be in JSON format as follows: {\"text\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="senior_editor",
            message=message,
            content_key="text",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        # print(prev_messages[-1])

        self.target_audience = content["text"]
        self.write_jsonl(save_path, [{"target_audience": content["text"]}])
        self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)

    def finalize_preparation(self):
        """
        finalize the preparation
        """
        print("*********************************************************************")
        print("********************** Finalizing the preparation... ****************")
        print("*********************************************************************")
        glossary_text = "\n".join([e["source"] + ": " + e["target"] for e in self.glossary])
        book_summary = self.book_summary
        tone = self.tone
        style = self.style
        target_audience = self.target_audience

        self.translation_guidelines = f"Glossary:\n\n{glossary_text}\n\nBook Summary:\n\n{book_summary}\n\nTone:\n\n{tone}\n\nStyle:\n\n{style}\n\nTarget Audience:\n\n{target_audience}"
        print(self.translation_guidelines)

    def translate(self):
        """
        translate the book
        """
        print("*********************************************************************")
        print("********************** Translating the book... **********************")
        print("*********************************************************************")
        translation_dir = os.path.join(self.project_save_dir, "translation")
        os.makedirs(translation_dir, exist_ok=True)

        num_chapters = len(self.book)
        for i in range(num_chapters):
            chapter_path = os.path.join(translation_dir, f"chapter_{i}.jsonl")
            if os.path.exists(chapter_path):
                print(f"Loading the translation of chapter {i} from {chapter_path}...")
                self.book[i]["chapter_translation_init"] = self.read_jsonl(chapter_path)[0]["chapter_translation_init"]
                self.book[i]["chapter_translation_init_length"] = self.read_jsonl(chapter_path)[0]["chapter_translation_init_length"]
            else:
                self.translate_one_chapter(i, chapter_path)

    def translate_one_chapter(self, chapter_idx, save_path):
        """
        translate one chapter
        """
        print(f"Translating chapter {chapter_idx}...")
        prev_messages = []

        curr_chapter = self.book[chapter_idx]
        chapter_title = curr_chapter["chapter_title"]
        chapter_text = curr_chapter["chapter_text"]

        translation_guidelines = self.translation_guidelines
        message = f"Translation Guidelines:\n\n{translation_guidelines}\n\nChapter Text:\n\n{chapter_text}\n\nTranslate the chapter text from {self.src_lang} into {self.tgt_lang}. Ensure that your translation closely adheres to the provided translation guidelines, including the glossary, book summary, tone, style, and target audience, for consistency and accuracy. Remember to maintain the original meaning and tone as much as possible while making the translation understandable in {self.tgt_lang}."
        additional_system_message = "Your response should always be in JSON format as follows: {\"translation\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="translator",
            message=message,
            content_key="translation",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        translation = content["translation"]
        translation_length = len(translation.split())

        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        prev_messages.append({"role": "translator", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["translator"]).write(content)

        message = f"Plese review the translation of chapter text, in terms of the glossary, book summary, tone, style, and target audience, and provide your suggestions for improvement."
        prev_messages.append({"role": "translator", "content": message})
        st.chat_message(self.project_roles["translator"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"suggestions\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=None,
            content_key="suggestions",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)

        # print(prev_messages[-1])

        # raise Exception("Stop here.")

        message = f"Please adjust the translation of chapter text if you think the translation can be improved."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"adjusted\": bool, \"translation\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="translator",
            message=None,
            content_key="translation",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        # print(content)
        # raise Exception("Stop here.")

        if content["adjusted"]:
            adjusted_translation = content["translation"]
            adjusted_translation_length = len(translation.split())

            ratio = adjusted_translation_length / translation_length
            print(adjusted_translation_length, translation_length, ratio)
            if ratio < 0.9:
                self.translate_one_chapter(chapter_idx, save_path)
                return None
        else:
            adjusted_translation = translation
            adjusted_translation_length = translation_length


        prev_messages.append({"role": "translator", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["translator"]).write(content)
        # print(prev_messages[-1])

        content, lst = self.evaluate_translation(chapter_text, adjusted_translation)
        prev_messages.extend(lst)
        if content["finalize"]:
            self.book[chapter_idx]["chapter_translation_init"] = adjusted_translation
            self.book[chapter_idx]["chapter_translation_init_length"] = adjusted_translation_length
            self.write_jsonl(save_path, [{"chapter_translation_init": adjusted_translation, "chapter_translation_init_length": adjusted_translation_length}])
            self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)
        else:
            self.translate_one_chapter(chapter_idx, save_path)
            return None

    def post_process(self):
        self.localize()
        self.proofread()
        self.finalize()
        self.write_down_the_book()
        
    def localize(self):
        """
        localize the book
        """
        print("*********************************************************************")
        print("********************** Localizing the book... ***********************")
        print("*********************************************************************")
        localization_dir = os.path.join(self.project_save_dir, "localization")
        os.makedirs(localization_dir, exist_ok=True)

        num_chapters = len(self.book)
        for i in range(num_chapters):
            chapter_path = os.path.join(localization_dir, f"chapter_{i}.jsonl")
            if os.path.exists(chapter_path):
                print(f"Loading the localization of chapter {i} from {chapter_path}...")
                self.book[i]["chapter_localization"] = self.read_jsonl(chapter_path)[0]["chapter_localization"]
                self.book[i]["chapter_localization_length"] = self.read_jsonl(chapter_path)[0]["chapter_localization_length"]
            else:
                self.localize_one_chapter(i, chapter_path)
        
    def localize_one_chapter(self, chapter_idx, save_path):
        """
        localize one chapter
        """
        print(f"Localizing chapter {chapter_idx}...")
        print(self.curr_rerun, self.max_rerun)
        if self.curr_rerun == self.max_rerun:
            
            adjusted_localization = self.book[chapter_idx]["chapter_translation_init"]
            adjusted_localization_length = self.book[chapter_idx]["chapter_translation_init_length"]

            self.book[chapter_idx]["chapter_localization"] = adjusted_localization
            self.book[chapter_idx]["chapter_localization_length"] = adjusted_localization_length
            self.write_jsonl(save_path, [{"chapter_localization": adjusted_localization, "chapter_localization_length": adjusted_localization_length, "remark": "reach max rerun"}])
            self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), [{"chapter_localization": adjusted_localization, "chapter_localization_length": adjusted_localization_length, "remark": "reach max rerun"}])
            return None

        prev_messages = []

        curr_chapter = self.book[chapter_idx]
        chapter_title = curr_chapter["chapter_title"]
        chapter_text = curr_chapter["chapter_text"]
        chapter_translation_init = curr_chapter["chapter_translation_init"]
        chapter_translation_init_length = curr_chapter["chapter_translation_init_length"]

        translation_guidelines = self.translation_guidelines
        message = f"Translation Guidelines:\n\n{translation_guidelines}\n\nChapter Text:\n\n{chapter_text}\n\nChapter Translation:\n\n{chapter_translation_init}\n\nGuided by our translation guidelines, including glossary, book summary, tone, style, and target audience, localize the chapter translation for {self.tgt_lang} context. You MUST maintain all the details and the orginal writing style of the chapter text."
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"localization\": string}. Please do not change the key of the JSON object. The \"localization\" key should be set to the localized chapter translation."
        local_content, response = self.call_api(
            assistant="localization_specialist",
            message=message,
            content_key="localization",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        # print(local_content)

        localization = local_content["localization"]
        localization_length = len(localization.split())

        ratio = localization_length / chapter_translation_init_length
        print(localization_length, chapter_translation_init_length, ratio)
        if ratio < 0.9:
            self.curr_rerun += 1
            self.localize_one_chapter(chapter_idx, save_path)
            return None

        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        prev_messages.append({"role": "localization_specialist", "content": json.dumps(local_content, ensure_ascii=False)})
        st.chat_message(self.project_roles["localization_specialist"]).write(local_content)
        # print(prev_messages[-1])

        message = f"Plese review the localized translation of chapter text, in terms of the glossary, book summary, tone, style, and target audience, and provide your suggestions for improvement. Please ensure that the localized translation is culturally adapted to the context of {self.tgt_lang}. Please also ensure that the localized translation is closely consistent with the original chapter text."
        prev_messages.append({"role": "localization_specialist", "content": message})
        st.chat_message(self.project_roles["localization_specialist"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"suggestions\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=None,
            content_key="suggestions",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)

        message = f"Please adjust the localized translation of chapter text accordingly if you think the translation can be improved."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)


        additional_system_message = "Your response should always be in JSON format as follows: {\"adjusted\": bool, \"localization\": string}. Please do not change the key of the JSON object. The value of \"adjusted\" should be set to false if the translation needs no adjustments. The \"localization\" key should be set to the adjusted localized chapter translation."
        content, response = self.call_api(
            assistant="translator",
            message=None,
            content_key="localization",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        # print(content)

        if content["adjusted"]:
            adjusted_localization = content["localization"]
            adjusted_localization_length = len(adjusted_localization.split())

            ratio = adjusted_localization_length / chapter_translation_init_length
            print(adjusted_localization_length, chapter_translation_init_length, ratio)
            if ratio < 0.9:
                self.curr_rerun += 1
                self.localize_one_chapter(chapter_idx, save_path)
                return None

        else:
            adjusted_localization = localization
            adjusted_localization_length = localization_length
        
        prev_messages.append({"role": "localization_specialist", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["localization_specialist"]).write(content)
        # print(prev_messages[-1])


        content, lst = self.evaluate_translation(chapter_text, adjusted_localization)
        prev_messages.extend(lst)
        if content["finalize"]:
            self.book[chapter_idx]["chapter_localization"] = adjusted_localization
            self.book[chapter_idx]["chapter_localization_length"] = adjusted_localization_length
            self.write_jsonl(save_path, [{"chapter_localization": adjusted_localization, "chapter_localization_length": adjusted_localization_length}])
            self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)
            return None
        else:
            self.curr_rerun += 1
            self.localize_one_chapter(chapter_idx, save_path)
            return None


        self.curr_rerun = 0
        return None

    def proofread(self):
        """
        proofread the book
        """
        print("*********************************************************************")
        print("********************** Proofreading the book... *********************")
        print("*********************************************************************")
        proofreading_dir = os.path.join(self.project_save_dir, "proofreading")
        os.makedirs(proofreading_dir, exist_ok=True)

        num_chapters = len(self.book)
        for i in range(num_chapters):
            chapter_path = os.path.join(proofreading_dir, f"chapter_{i}.jsonl")
            if os.path.exists(chapter_path):
                print(f"Loading the proofreading of chapter {i} from {chapter_path}...")
                self.book[i]["chapter_proofreading"] = self.read_jsonl(chapter_path)[0]["chapter_proofreading"]
            else:
                self.proofread_one_chapter(i, chapter_path)
    
    def proofread_one_chapter(self, chapter_idx, save_path):
        """
        proofread one chapter
        """
        print(f"Proofreading chapter {chapter_idx}...")
        print(self.curr_rerun, self.max_rerun)
        if self.curr_rerun == self.max_rerun:
            adjusted_proofreading = self.book[chapter_idx]["chapter_localization"]
            adjusted_proofreading_length = self.book[chapter_idx]["chapter_localization_length"]

            self.book[chapter_idx]["chapter_proofreading"] = adjusted_proofreading
            self.book[chapter_idx]["chapter_proofreading_length"] = adjusted_proofreading_length
            self.write_jsonl(save_path, [{"chapter_proofreading": adjusted_proofreading, "chapter_proofreading_length": adjusted_proofreading_length, "remark": "reach max rerun"}])
            self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), [{"chapter_proofreading": adjusted_proofreading, "chapter_proofreading_length": adjusted_proofreading_length, "remark": "reach max rerun"}])
            return None

        prev_messages = []

        curr_chapter = self.book[chapter_idx]
        chapter_title = curr_chapter["chapter_title"]
        chapter_text = curr_chapter["chapter_text"]
        # chapter_translation_init = curr_chapter["chapter_translation_init"]
        chapter_localization = curr_chapter["chapter_localization"]
        chapter_localization_length = curr_chapter["chapter_localization_length"]

        translation_guidelines = self.translation_guidelines
        message = f"Translation Guidelines:\n\n{translation_guidelines}\n\nChapter Text:\n\n{chapter_text}\n\nChapter Translation:\n\n{chapter_localization}\n\nGuided by our translation guidelines, including the glossary, book summary, tone, style, and target audience, proofread the chapter translation."
        additional_system_message = "Your response should always be in JSON format as follows: {\"proofreading\": string}. Please do not change the key of the JSON object. The \"proofreading\" key should be set to the proofread chapter translation."
        proof_content, response = self.call_api(
            assistant="proofreader",
            message=message,
            content_key="proofreading",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )

        proofreading = proof_content["proofreading"]
        proofreading_length = len(proofreading.split())

        ratio = proofreading_length / chapter_localization_length
        print(proofreading_length, chapter_localization_length, ratio)
        if ratio < 0.9:
            self.curr_rerun += 1
            self.proofread_one_chapter(chapter_idx, save_path)
            return None

        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        prev_messages.append({"role": "proofreader", "content": json.dumps(proof_content, ensure_ascii=False)})
        st.chat_message(self.project_roles["proofreader"]).write(proof_content)

        message = f"Plese review the proofread translation of chapter text, in terms of the glossary, book summary, tone, style, and target audience, and provide your suggestions for improvement."
        prev_messages.append({"role": "proofreader", "content": message})
        st.chat_message(self.project_roles["proofreader"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"suggestions\": string}. Please do not change the key of the JSON object."
        content, response = self.call_api(
            assistant="junior_editor",
            message=None,
            content_key="suggestions",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        prev_messages.append({"role": "junior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["junior_editor"]).write(content)

        message = f"Please adjust the proofread translation of chapter text accordingly if you think the translation can be improved."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)

        additional_system_message = "Your response should always be in JSON format as follows: {\"adjusted\": bool, \"proofreading\": string}. Please do not change the key of the JSON object. The value of \"adjusted\" should be set to false if the translation needs no adjustments. The \"proofreading\" key should be set to the adjusted proofread chapter translation."
        content, response = self.call_api(
            assistant="proofreader",
            message=None,
            content_key="proofreading",
            additional_system_message=additional_system_message,
            prev_messages=prev_messages,
        )
        # print(content)

        if content["adjusted"]:
            adjusted_proofreading = content["proofreading"]
            adjusted_proofreading_length = len(adjusted_proofreading.split())

            ratio = adjusted_proofreading_length / chapter_localization_length
            print(adjusted_proofreading_length, chapter_localization_length, ratio)
            if ratio < 0.9:
                self.curr_rerun += 1
                self.proofread_one_chapter(chapter_idx, save_path)
                return None
        else:
            adjusted_proofreading = proofreading
            adjusted_proofreading_length = proofreading_length

        prev_messages.append({"role": "proofreader", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["proofreader"]).write(content)
        # print(prev_messages[-1])

        content, lst = self.evaluate_translation(chapter_text, adjusted_proofreading)
        prev_messages.extend(lst)
        if content["finalize"]:
            self.book[chapter_idx]["chapter_proofreading"] = adjusted_proofreading
            self.book[chapter_idx]["chapter_proofreading_length"] = adjusted_proofreading_length
            self.write_jsonl(save_path, [{"chapter_proofreading": adjusted_proofreading, "chapter_proofreading_length": adjusted_proofreading_length}])
            self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)
            return None
        else:
            self.curr_rerun += 1
            self.proofread_one_chapter(chapter_idx, save_path)
            return None

        self.curr_rerun = 0
        return None

    def finalize(self):
        """
        finalize the book
        """
        print("*********************************************************************")
        print("********************** Finalizing the book... ***********************")
        print("*********************************************************************")
        finalization_dir = os.path.join(self.project_save_dir, "finalization")
        os.makedirs(finalization_dir, exist_ok=True)


        num_chapters = len(self.book)
        for i in range(num_chapters):
            chapter_path = os.path.join(finalization_dir, f"chapter_{i}.jsonl")
            if os.path.exists(chapter_path):
                print(f"Loading the finalization of chapter {i} from {chapter_path}...")
                self.book[i]["chapter_finalization"] = self.read_jsonl(chapter_path)[0]["chapter_finalization"]
            else:
                redo_outcome = self.finalize_one_chapter(i, chapter_path)
                if redo_outcome is not None:
                    self.redo_one_chapter()

    def finalize_one_chapter(self, chapter_idx, save_path):
        """
        finalize one chapter
        """
        print(f"Finalizing chapter {chapter_idx}...")
        prev_messages = []

        curr_chapter = self.book[chapter_idx]
        chapter_title = curr_chapter["chapter_title"]
        chapter_text = curr_chapter["chapter_text"]
        # chapter_translation_init = curr_chapter["chapter_translation_init"]
        # chapter_localization = curr_chapter["chapter_localization"]
        chapter_translation = curr_chapter["chapter_proofreading"]

        if chapter_idx == 0:
            prev_chapter_translation = ""
        else:
            prev_chapter_translation = self.book[chapter_idx-1]["chapter_proofreading"]

        translation_guidelines = self.translation_guidelines

        message = f"Translation Guidelines:\n\n{translation_guidelines}\n\nPrevious Chapter Translation:\n\n{prev_chapter_translation}\n\nCurrent Chapter Text\n\n{chapter_text}\n\nCurrent Chapter Translation:\n\n{chapter_translation}\n\nConsidering the translation guidelines, including the glossary, book summary, tone, style, and target audience, please review if the current chapter aligns well with the previous chapter translation and the current chapter text. This is the final step before the chapter is considered complete, so you must ensure that the current chapter translation is error-free."
        prev_messages.append({"role": "junior_editor", "content": message})
        st.chat_message(self.project_roles["junior_editor"]).write(message)
        additional_system_message = "Your response should always be in JSON format as follows: {\"justification\": string, \"finalize\": bool}. The value of \"finalize\" should be set to true if the current chapter aligns with the previous chapter. Please do not change the key of the JSON object."
        # print(message)
        content, response = self.call_api(
            assistant="senior_editor",
            message=message,
            content_key="finalize",
            additional_system_message=additional_system_message,
            prev_messages=[],
        )
        print(content)
        # raise Exception("Stop here.")
        prev_messages.append({"role": "senior_editor", "content": json.dumps(content, ensure_ascii=False)})
        st.chat_message(self.project_roles["senior_editor"]).write(content)
        if content["finalize"]:
            self.book[chapter_idx]["chapter_finalization"] = chapter_translation
            self.write_jsonl(save_path, [{"chapter_finalization": chapter_translation}])
            self.write_jsonl(save_path.replace(".jsonl", "_conv.jsonl"), prev_messages)
            return None
        else:
            return chapter_idx

    def redo_one_chapter(self, chapter_idx):
        """
        redo one chapter
        """
        print(f"Redoing chapter {chapter_idx}...")
        redo_dir = os.path.join(self.project_save_dir, "redo")
        os.makedirs(redo_dir, exist_ok=True)
        chapter_path = os.path.join(redo_dir, f"chapter_{chapter_idx}_translation.jsonl")
        self.translate_one_chapter(chapter_idx, chapter_path)

        chapter_path = os.path.join(redo_dir, f"chapter_{chapter_idx}_localization.jsonl")
        self.localize_one_chapter(chapter_idx, chapter_path)

        chapter_path = os.path.join(redo_dir, f"chapter_{chapter_idx}_proofreading.jsonl")
        self.proofread_one_chapter(chapter_idx, chapter_path)

        chapter_path = os.path.join(redo_dir, f"chapter_{chapter_idx}_finalization.jsonl")
        redo_outcome = self.finalize_one_chapter(chapter_idx, chapter_path)
        if redo_outcome is not None:
            self.redo_one_chapter(chapter_idx)
        
        return None

    def write_down_the_book(self):
        """
        write down the book
        """
        print("*********************************************************************")
        print("********************** Writing down the book... *********************")
        print("*********************************************************************")
        book_path = os.path.join(self.project_save_dir, "book.jsonl")
        self.write_jsonl(book_path, self.book)
    



    def call_api(self, assistant, message, content_key, additional_system_message=None, prev_messages=[]):
        """
        call the API to translate the text
        """
        # print(additional_system_message)
        def update_role_prev_messages(assistant_role, prev_messages):
            new_prev_messages = []
            for m in prev_messages:
                if m["role"] == assistant_role:
                    new_prev_messages.append(
                        {"role": "assistant", "content": m["content"]}
                    )
                else:
                    new_prev_messages.append(
                        {"role": "user", "content": m["content"]}
                    )
            return new_prev_messages


        time.sleep(1)
        # call_api_uuid = str(uuid.uuid4())

        model = self.project_members[assistant]["model"]
        role_prompt = self.project_members[assistant]["role_prompt"]

        messages = [
            {"role": "system", "content": self.company_prompt},
            {"role": "system", "content": role_prompt},
        ]
        if additional_system_message is not None:
            messages.append({"role": "system", "content": additional_system_message})

        if len(prev_messages) > 0:
            prev_messages = update_role_prev_messages(assistant, prev_messages)
            messages.extend(prev_messages)

        if message is not None:
            messages.append({"role": "user", "content": message})

        # for m in messages:
        #     print(m)
        # print("===================")

        retry = 0
        flag = False
        raw_response = None
        while retry < self.max_retry:
            try:
                raw_response = self.client.chat.completions.create(
                    model=model,
                    response_format={ "type": "json_object" },
                    messages=messages,
                    temperature=0.7,
                )
                content = json.loads(raw_response.choices[0].message.content)
                output = content[content_key]
                # print("========", content)
                # print(content_key, content.keys())
                if not content_key in content.keys():
                    raise Exception(f"Failed to get the content key {content_key} from the response.")
                break

            except Exception as e:
                print(e)
                print(raw_response)
                retry += 1
                print(f"Retry {retry} times for calling api...")
                time.sleep(1)

        return content, raw_response.model_dump()





def main():
    # parser = argparse.ArgumentParser()

    langs = ("Chinese", "English")
    st.title("TransChat")
    with st.sidebar:
        api_key = st.text_input("Your api key")
        src_lang = st.selectbox("source language", langs)
        tgt_lang = st.selectbox("target language", langs)
        uploaded_file = st.file_uploader("Your file")

        num_senior_editors = st.slider("Number of Senior Editors", 1, 5, 2) 
        num_junior_editors = st.slider("Number of Junior Editors", 1, 5, 2) 
        num_translators = st.slider("Number of Translators", 1, 5, 2) 
        num_localization_specialists = st.slider("Number of Localization Specialists", 1, 5, 2) 

        num_proofreaders = st.slider("Number of Proofreaders", 1, 5, 2) 
        num_beta_readers = st.slider("Number of Beta Readers", 1, 5, 2) 
        max_turns= st.slider("Number of Max Converstaion Turns", 1, 10, 3) 
        max_retry=st.slider("Number of Maximum Retry", 1, 10, 3)
        max_rerun=st.slider("Number of Maximum Return", 1, 10, 5) 

    if not os.path.exists("output"):
        os.makedirs("output")

    # st.columns(1)
    if st.button('Start Processing') and api_key is not None and uploaded_file is not None:

        file_details = {
            "filename": uploaded_file.name,
            "content": uploaded_file.getvalue()
        }
        save_path = os.path.join("uploads", file_details["filename"])
        with open(save_path, "wb") as f:
            f.write(file_details["content"])


        client = OpenAI(api_key=api_key)

        chat = TransChat(
            client=client,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            text_path=save_path,
            save_dir="output",
            num_senior_editors=num_senior_editors, 
            num_junior_editors=num_junior_editors,
            num_translators=num_translators, 
            num_localization_specialists=num_localization_specialists, 
            num_proofreaders=num_proofreaders,
            num_beta_readers=num_beta_readers,
            max_turns=max_turns,
            max_retry=max_retry,
            max_rerun=max_rerun,
        )

        chat.execute()


if __name__=="__main__":
    main()

