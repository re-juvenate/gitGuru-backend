import json

import time
import yaml
from typing import Any, Optional, List, Mapping, Dict


from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

from ai import llmloader

# Constants
config = {}
timestr = lambda: time.strftime("%Y-%m-%d")
with open("settings.yaml", "r") as f:
    config = yaml.safe_load(f)
llmloader.set_opts(config)

to_json_prompt = PromptTemplate.from_template(
    """Parse the given input's structure and ONLY output a JSON based on that structure. Output only the JSON in curly brackets, NOTHING else.
    Input: 
    {input}"""
)
json_corrector_prompt = PromptTemplate.from_template(
    """Correct the given input's structure and ONLY output valid corrected JSON. Output ONLY the JSON in curly brackets. Shut up and do the work.
    Input: 
    {input}"""
)


def llm_to_json(input):
    ollama_llm = llmloader.load_llm(provider="ollama")
    jsoner = to_json_prompt | ollama_llm | StrOutputParser()
    json_corrector = json_corrector_prompt | ollama_llm | StrOutputParser()
    x = jsoner.invoke({"input": input})
    x = json_corrector.invoke({"input": x})
    # x = json.loads(x)
    return x


def chunk(start_index=0, text_messages: List[str] = [], n=500):
    word_count = 0
    d_index = 0
    for d_index, msg in enumerate(text_messages[start_index:]):
        word_count += len(msg.split())
        if word_count > n:
            word_count = 0
            break

    return start_index + d_index, text_messages[start_index : start_index + d_index]


def fmt(messages: List[str]) -> str:
    return "\n".join(messages)


@chain
def msg_summarizer(inputs: Dict):
    summarizer_prompt = PromptTemplate.from_template(
        """Summarize the given developers' comments on github repo {repo_info}, preserving only the important meaning and details into around 150-200 words.
        Commments:
        {input}"""
    )
    ollama_llm = llmloader.load_llm(provider="ollama")
    summ = summarizer_prompt | ollama_llm | StrOutputParser()
    end_index = 0
    summary = ""
    while end_index < len(inputs["messages"])-1:
        end_index, thread_msgs = chunk(end_index, inputs["messages"])
        msgs = [summary]
        msgs.extend(thread_msgs)
        text = fmt(msgs)
        summary = summ.invoke({"input": text, "repo_info": inputs["repo_info"]})
    return summary


if __name__ == "__main__":
    repo_info = "internetarchive/openlibrary"
    # sample_data = [
    #     "can i work on this?\r\n",
    #     "@mekarpeles Hey sir, I would like to work on it.",
    #     "@nick2432, I have assigned this to you. Sorry for the delay and please let me know if you have any questions.\r\n\r\n@shivam200446, maybe one of these might be a good fit? https://github.com/internetarchive/openlibrary/issues?q=is%3Aopen+label%3A%22Good+First+Issue%22+no%3Aassignee.",
    #     "Hey, @nick2432, are you working on this? If not, I can help with this issue.",
    #     "Hi @scottbarnes, I did some research and it seems that our task is to verify whether the first 100,000 most cited books from Opnealex are available in the ol or not. The parameters that I have used for this task are title, author, and publish_year. Do you think there are any other parameters that should be taken into consideration? I have written a Python script to record all the missing books in a file.",
    #     "I ran the program for the first 10,000 books and discovered the following:\r\n1. Out of the 10,000 books, 2,194 had no results in the OL search.\r\n2. There are issues with decoding JSON which is causing failures in many cases\r\n3. Including the 2nd case, and some Connection aborted errors. A total of 663 books didn't get processed due to these errors.\r\n\r\nOnce I receive @scottbarnes  inputs, I'll try to fix the errors and run the program for all the books.",
    #     "@Billa05, I followed up on Slack. :)",
    #     "Hey @scottbarnes , I've shared the link to the code on Slack.",
    #     "@Billa05, I ran the code on a few hundred items, but didn't get any errors. However, I left some feedback on the gist you shared that may help explain where the errors are coming from.",
    #     "Yeah @scottbarnes, I did the same and it's working perfectly. Maybe I was having network problems that day. what would be the next step towards this issue?",
    #     "@Billa05, I think the next step is to simply report what kind of matches turn up. For example, perhaps see how many matches there are when you try searching with whatever information is available (e.g. if there is no author, still trying a search and seeing if there is a match). Then trying again but requiring both an author and a title. And finally maybe requiring all three match.\r\n\r\nWhat else sounds like it might be interesting to see?",
    #     "Hey @scottbarnes, I can think of 4 cases: \r\n1) only title\r\n2) title and author\r\n3) title and publish date\r\n4) all three parameters.\r\n\r\nFor now, I will only take the first 300 books. Once done, I will share the record in Google Docs.",
    #     "hey @scottbarnes I experimented and here's what I got :\r\n\r\nnumber of books found in OL (out of first 300 books):\r\n\r\n- Only title: 272 books\r\n- Title & publish_date: 272 books\r\n- Title & author: 241 books\r\n- All three parameters: 241 books\r\n\r\nIt appears that the first two and the last two yield identical outcomes.\r\n",
    #     "Assignees removed automatically after 14 days.",
    #     "Hi, @Billa05. Thanks for this. I will talk with @mekarpeles about how to proceed.\r\n\r\nOne thing I realize I may not have been clear about is I was wondering, for example, how many _only_ had a title, and how many _only_ had a title and publish_date. I know it could be a coincidence that they're the same, as are the other two, just to make sure, there are 272 with _only_ a title, correct, and a different 272 with _only_ a title and publish_date?",
    #     "Open Alex has metadata, including ISBN, we want to find e.g. the top 100k (or 10k) books w/ ISBN.\r\n\r\nWe want to look for these 10k or 100k isbns in e.g. Open Library data dump or search (in batches of ~100) and see which Open Library records exist v. are missing.\r\n\r\nCompile a list of all of the hits (e.g. `openalex_id`, `isbn`, `openlibrary_edition`) and keep track of misses (e.g. `openalex_id`, `isbn`).\r\n\r\n**Next steps:**\r\nEventually, we want (a) import any missing record, (b) add `openalex` identifiers for any existing record, (c) give the openalex folks a query so they can see all Open Library records with `openalex` identifiers and also `ocaids` (meaning they have previews or are readable/borrowable.\r\n",
    # ]
    get_issue_comment("internetarchive","openlibrary",8623)
    a = msg_summarizer.invoke(
        {
            "messages": sample_data,
            "repo_info": repo_info,
        }
    )
    print(a)
    pass
