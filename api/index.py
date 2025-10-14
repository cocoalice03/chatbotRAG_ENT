from vercel_wsgi import handle
from main import app as flask_app


def handler(request, *args, **kwargs):
    return handle(request, flask_app)
