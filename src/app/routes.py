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
    print(rel_data)
    response = chains.explain_issue(issue, rel_data)
    return response


@router.post("/summ_msgs/")
async def summarize_comments(repo: models.Repo):
    data = functions.url_parser(repo.url)
    response = functions.get_issue_comment(*data)
    response = chains.summ(response)
    # response = functions.get_issue_comment("internetarchive","openlibrary",8623)
    return {"summary": response}


@router.post("/find_sols/")
async def find_solns(repo: models.Repo):
    data = functions.url_parser(repo.url)
    response = "Null"
    return response
