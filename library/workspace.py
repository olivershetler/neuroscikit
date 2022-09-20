from core.session import Session

class Workspace():
    def __init__(self):
        pass


class SessionWorkspace(Workspace):
    def __init__(self, Session):
        self.session_id = session_id



class StudyWorkspace(Workspace):
    def __init__(self, *args):
        for arg in args:
            assert isinstance(arg, SessionWorkspace), 'All arguments must be SessionWorkspace objects'
        self.sessions = list(args)
    def add_session(self, session:SessionWorkspace):
        assert isinstance(session, SessionWorkspace), 'The argument must be a SessionWorkspace object'
        self.sessions.append(session)