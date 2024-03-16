from fastapi import APIRouter

from fastapi import HTTPException
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from app import models
from app import functions
from ai import chains


router = APIRouter()


@router.post("/explain_issue/")
async def explain_issue(repo: models.Repo):
    data = functions.url_parser(repo.url)
    issue = functions.get_issue_body(*data)
    rel_data = functions.get_repo_readme(*data)
    response = chains.explain_issue(issue, rel_data)
    return {"text": response}


@router.post("/summ_msgs/")
async def summarize_comments(repo: models.Repo):
    data = functions.url_parser(repo.url)
    response = functions.get_issue_comment(*data)
    response = chains.summ(response)
    # response = functions.get_issue_comment("internetarchive","openlibrary",8623)
    return {"text": response}


@router.post("/find_sols/")
async def find_solns(repo: models.Repo):
    data = functions.url_parser(repo.url)
    title = functions.get_issue_title(*data)
    full = functions.get_issue_body(*data)
    langs = list(functions.get_repo_language(*data).keys())
    code = functions.get_issue_code(*data)
    repo_path = data[0] + "/" + data[1]
    response = chains.get_possible_solns(title, full, repo_path, langs, code)

    return response
