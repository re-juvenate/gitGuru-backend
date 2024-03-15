from fastapi import APIRouter

from fastapi import HTTPException
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from app import functions
from ai import chains


router = APIRouter()

DATA_STORE_PATH = "data/user_doc"

@router.post("/explain_issue/")
async def get_issue_url(url: str):
    data = functions.url_parser(url)
    response = "Null"
    return response

@router.post("/summ_msgs/")
async def summarize_comments(url: str):
    data = functions.url_parser(url)
    response = functions.get_issue_comment(*data)
    response = chains.summ(response)
    # response = functions.get_issue_comment("internetarchive","openlibrary",8623)
    return {"summary":response}

@router.post("/find_sols/")
async def find_solns(url: str):
    data = functions.url_parser(url)
    response = "Null"
    return response
