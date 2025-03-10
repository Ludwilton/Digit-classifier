from main import server  # main.py should define 'server = app.server'
from werkzeug.wrappers import Request, Response

def handler(request):
    return Response.from_app(server, request.environ)