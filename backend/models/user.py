from pydantic import BaseModel, EmailStr
from typing import Optional


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    is_approved: Optional[bool] = False

    class Config:
        from_attributes = True
