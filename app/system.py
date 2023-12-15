import importlib
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from jose import jwt
from pydantic import BaseModel

from app.pipelines import load_default_pipeline

router = APIRouter()

current_token = None
last_access_time = None

ALGORITHM = "HS256"
SECRET_KEY = os.environ.get("SECRET_KEY", "secret key")

pipeline, default_params = load_default_pipeline()


class PipelineRequest(BaseModel):
    pipeline: str
    build_args: Optional[dict] = {}


def validate_token(authorization: str | None = Header(), token: str | None = None):
    global current_token, last_access_time

    if current_token is None or last_access_time is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No token available"
        )

    # 检查令牌是否匹配
    if (token is not None and token != current_token) or (
        token is None and authorization != f"Bearer {current_token}"
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    # 检查令牌是否过期
    if datetime.now() - last_access_time > timedelta(minutes=1):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )


def generate_token():
    # 设置JWT令牌的过期时间
    expire_time = datetime.utcnow() + timedelta(minutes=1)

    # 构建JWT令牌的payload
    payload = {"sub": "muse_lcm", "exp": expire_time}

    # 生成JWT令牌
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


def refresh_access_time():
    global last_access_time
    # 更新最后一次访问时间
    last_access_time = datetime.now()


@router.get("/token")
async def get_token():
    global current_token, last_access_time

    # 无token或上次操作已经经过1分钟
    if current_token is None or datetime.now() - last_access_time > timedelta(
        minutes=1
    ):
        # 生成新的令牌
        current_token = generate_token()
        refresh_access_time()
        return {"token": current_token}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="No token available"
    )


@router.get("/refresh", dependencies=[Depends(validate_token)])
async def refresh_token():
    refresh_access_time()

    return {"message": "success"}


@router.post("/disconnect", dependencies=[Depends(validate_token)])
async def invalidate_token():
    global current_token, last_access_time
    current_token = None
    last_access_time = None
    return {"message": "success"}


@router.post("/update_pipeline")
async def update_pipeline(req: PipelineRequest):
    global pipeline, default_params

    try:
        pipeline_module = importlib.import_module(f"app.pipelines.{req.pipeline}")
        pipeline, default_params = pipeline_module.build_pipeline(req.build_args)

        return {"status": "ok"}
    except Exception as e:
        print(f"failed to update pipeline {req.pipeline}: {e}")
