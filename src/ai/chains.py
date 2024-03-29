import json

import time
import yaml
from typing import Any, Optional, List, Mapping, Dict


from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    TokenTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain_community.utilities.searx_search import SearxSearchWrapper

from ai import llmloader
from ai.clusterer import cluster

# Constants
datestr = lambda: time.strftime(r"%Y-%m-%d")
config = {}
with open("settings.yaml", "r") as f:
    config = yaml.safe_load(f)
llmloader.set_opts(config)

ollama_llm = llmloader.load_llm(provider="ollama")
ds_llm = llmloader.load_llm(provider="fireworks")
ollama_embed = llmloader.load_ollama_embed()


def llm_to_json(input):
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
    jsoner = to_json_prompt | ollama_llm | StrOutputParser()
    json_corrector = json_corrector_prompt | ollama_llm | StrOutputParser()
    x = jsoner.invoke({"input": input})
    x = json_corrector.invoke({"input": x})
    # x = json.loads(x)
    return x


def fmt(messages: List[str]) -> str:
    return "---\n".join(messages)


def summ(msgs):
    summ_prompt = PromptTemplate.from_template(
        """Summarize the developers' comments and remarks clearly while preserving all the specific details, technical terms, (@usernames) and important meaning into a few extremely detailed points. Remove unimportant greetings, formalities and thank-you's if needed.
        Dev. Comments: 
        {input}
        
        """
    )
    consistency_prompt = PromptTemplate.from_template(
        """Paraphrase the developers' comments and remarks into a consistent flow. Preserve all specifc details, technical terms and @usernames into a few concise, logically consistent paragraphs.
        Comments:
        {input}

        """
    )
    summ_er = summ_prompt | ollama_llm | StrOutputParser()
    const_er = consistency_prompt | ollama_llm | StrOutputParser()

    if len(msgs) > 6:
        texts = cluster(msgs, ollama_embed)
    else:
        texts = msgs
    texts = [{"input": text} for text in texts]
    summ_s = summ_er.batch(texts, config={"max_concurrency": 6})
    summ_s = fmt(summ_s)
    summary = const_er.invoke({"input": summ_s})
    return summary


def cluster_sums(docs, s=2):
    summ_prompt = PromptTemplate.from_template(
        """Summarize the given information concisely while preserving all the details, unknown terms, (@usernames) and important meaning into a few detailed points. Remove any verbosity and unimportant greetings, formalities and thank-you's as needed.
        Input:
        {input}

        """
    )
    summ_er = summ_prompt | ollama_llm | StrOutputParser()
    if len(docs) > 6:
        texts = cluster(docs, ollama_embed, min_s=s)
    else:
        texts = docs
    texts = [{"input": text} for text in texts]
    summ_s = summ_er.batch(texts, config={"max_concurrency": 6})
    return summ_s


def batch_sums(texts):
    summ_prompt = PromptTemplate.from_template(
        """Summarize the given information concisely while preserving all the important meaning, details, unknown terms, (@usernames) into a few detailed points. Remove any verbosity and unimportant greetings, formalities, and thank-you's as needed.
        Input:
        {input}

        """
    )
    summ_er = summ_prompt | ollama_llm | StrOutputParser()

    texts = [{"input": text} for text in texts]
    summ_s = summ_er.batch(texts, config={"max_concurrency": 6})
    return summ_s


def conster(summ_s):
    consistency_prompt = PromptTemplate.from_template(
        """Paraphrase the given input into a consistent flow. Preserve all details, technical terms and jargon into a few concise, logically consistent paragraphs.
        Comments:
        {input}

        """
    )
    const_er = consistency_prompt | ollama_llm | StrOutputParser()
    summary = const_er.invoke({"input": fmt(summ_s)})
    return summary


def explain_issue(issue, related_text):
    rel_text_docs = TokenTextSplitter(chunk_size=128, chunk_overlap=32).split_text(
        related_text
    )
    rel_text = cluster_sums(rel_text_docs)
    print(rel_text)
    rel_text = conster(rel_text)

    expl_prompt = PromptTemplate.from_template(
        """Please explain the given Github issue concisely in points. WITHOUT skipping any important details.
        Github Issue:
        {input}
        Github repository details(you can use this as context, but DO NOT write this part):
        {related_text} 

        Requirements:
        * Explain what the issue is clearly 
        * Explain the reason for the issue
        * Mention some important details, such as the poster's setup, steps to reproduce etc. mentioned in the issue concisely.
        Tips: Use clean, concise, consistent formatting to get your points across.
        """
    )
    expl_chain = expl_prompt | ds_llm | StrOutputParser()
    explanation = expl_chain.invoke({"input": issue, "related_text": rel_text})
    return explanation


def code_solutions(
    repo,
    issue,
    issue_title,
    code,
    langs,
):

    search = SearxSearchWrapper(searx_host=config["searx_url"])
    results = search.results(
        issue_title + ", ".join(langs[:5]),
        engines=[
            "google",
            "presearch",
            "bing",
            "stackoverflow",
            "github",
            "gitlab",
            "reddit",
        ],
        num_results=8,
    )

    results = [res["title"] + " " + res["snippet"] for res in results]
    results = batch_sums(results)

    code = fmt(code)
    results = fmt(results)
    langs = ", ".join(langs[:5])

    coder_prompt = PromptTemplate.from_template(
        """You have to fix a GitHub issue on {repo}. Write clean, understandable code and be brief: 
        {issue}
        using {langs}.
        Others have written and tried the given code snippets below in. Keeping that in mind, program a fix for the issue.
        Code snippets:
        {code}
        Use the given info to guide your decisions:
        {info}

        Only write a single block of code that fixes the issue. Nothing else is needed.
    """
    )

    codeer = coder_prompt | ds_llm | StrOutputParser()
    suggestions = codeer.invoke(
        {"repo": repo, "issue": issue, "langs": langs, "code": code, "info": results},
    )
    return suggestions


def expl_solutions(issue_full, repo, soln, code):

    sol_prompt = PromptTemplate.from_template(
        """
        You are a developer who has solved this issue in the repo {repo} in your free time: 
        {issue}
        using the given code:
        Your solution: 
        {soln}
        Other possible solutions:
        {code}
        Write a solution/guide to the solution, nothing else is needed.
        Write on how you resolved the issue in one line.
        """
    )

    sol_chain = sol_prompt | ds_llm | StrOutputParser()
    final_result = sol_chain.invoke(
        {
            "repo": repo,
            "issue": issue_full,
            "soln": soln,
            "code": fmt(code),
        }
    )
    return final_result


def ffr(
    repo,
    files,
    issue,
    langs,
):

    langs = ", ".join(langs[:5])

    p1 = PromptTemplate.from_template(
        """
    The program repository {repo} is coded in {langs}.
    However, there is an issue in the code:
    {issue}
    Files in the repo:
    {files}
    What language do you think will the solution be in?
    Choose one from [{alllangs}] (ONLY output a single word, NOTHING ELSE):
    """
    )

    p2 = PromptTemplate.from_template(
        """You are debugging an issue in a repo {repo} programmed in {langs}.
    Issue: {issue}

    The following file paths are present in the repo: {files}

    Which the given files may contain the relevant functions/code for solving this issue?
    Only output one file path, in ONE line,NO explanation, just a single file path, NOTHING ELSE.

    File path: """
    )

    p4 = PromptTemplate.from_template(
        """
    You are an expert linux sysadmin. Your current location is {repo}:
    The following subdir structure is present:
    {files}
    Describe the structure in natural language. Make sure to mention filenames and paths and other details.
    """
    )
    all_langs = ", ".join([e.value for e in Language])

    lfinder = (
        p1.partial(langs=langs, issue=issue, alllangs=all_langs)
        | ds_llm
        | StrOutputParser()
    )
    pfinder = p2.partial(langs=langs, issue=issue) | ds_llm | StrOutputParser()
    desc_fs = p4 | ollama_llm | StrOutputParser()

    fftool = RunnableParallel(lfinder=lfinder, pfinder=pfinder, desc_fs=desc_fs)
    fdet = fftool.invoke({"repo": repo, "files": files})
    return fdet


def progger(repo, title, code, lfinder, pfinder, desc_fs):
    # search = SearxSearchWrapper(searx_host=config["searx_url"])
    # results = search.results(
    #     title + ", ".join(langs[:5]),
    #     engines=[
    #         "stackoverflow",
    #         "github",
    #         "gitlab",
    #         "reddit",
    #     ],
    #     num_results=8,
    # )

    # results = [res["title"] + " " + res["snippet"] for res in results]
    # results = batch_sums(results)
    # res = conster(results)
    # results = fmt(results)
    # code = fmt(code)
    spl = RecursiveCharacterTextSplitter.get_separators_for_language(lfinder.strip().split()[0])
    spl = RecursiveCharacterTextSplitter(separators=spl)
    code = spl.split_text(code)
    code_vecstore = FAISS.from_texts(code, embedding=ollama_embed)
    retriever = code_vecstore.as_retriever(k=4)
    p3 = PromptTemplate.from_template(
        """You need to fix an issue in {repo} (probably) in {path}
        Issue: {issue}
        File Structure: {desc_fs}
        Code blocks: {code} 
        Refactor the code to rectify this issue and review your code to ensure its validity.
    """
    )
    refactorer = (
        {
            "code": retriever,
            "repo": RunnablePassthrough(),
            "path": RunnablePassthrough(),
            "issue": RunnablePassthrough(),
            "desc_fs": RunnablePassthrough(),
        }
        | p3
        | ds_llm
        | StrOutputParser()
    )
    # You can also see the separators used for a given language
    ans = refactorer.invoke(
        {
            "desc_fs": desc_fs,
            "path": pfinder,
            "repo": repo,
            "issue": title,
        }
    )
    return ans


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
    with open("data/Comment.txt", "r") as f:
        sample_data = f.read()
    sample_data = eval(sample_data)
    a = summ(sample_data)
    print(len(sample_data), len(a))
    # get_issue_comment("internetarchive", "openlibrary", 8623)
    print(a)
    pass
