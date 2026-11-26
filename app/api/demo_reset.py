from fastapi import APIRouter, HTTPException, Request, Response, status

from app.core.logging import get_logger
from app.services.demo_reset import DemoResetError, DemoResetService

router = APIRouter(tags=["demo"], include_in_schema=False)
logger = get_logger("api.demo_reset")


@router.delete("/demo/data", status_code=status.HTTP_204_NO_CONTENT)
def restart_demo(request: Request) -> Response:
    settings = request.app.state.settings
    if settings.app_env != "development":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    try:
        DemoResetService(settings.recordings_path, settings.fixture_path).reset()
    except DemoResetError as error:
        logger.error(
            "Could not clear the local demo data. | event=demo_reset_failed error_type=%s",
            type(error.__cause__).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not clear demo data",
        ) from error
    logger.info(
        "Cleared the local demo data; the demo is ready to restart. | event=demo_reset_completed"
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
