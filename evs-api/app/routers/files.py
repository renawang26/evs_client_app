from fastapi import APIRouter, Depends
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()


@router.get("")
def list_files(_: User = Depends(get_current_user)):
    """Stub endpoint — returns empty list. Real implementation in Task 7."""
    return []
